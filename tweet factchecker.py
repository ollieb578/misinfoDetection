import numpy as np
import pandas as pd
import tensorflow as tf
import transformers
import string
import timeit
import os

dataset_name = 'semantic'
saved_model_path = './{}_bert'.format(dataset_name.replace('/', '_'))

batch_size = 32
max_length = 128
raw_labels = ["contradiction", "entailment", "neutral"]
final_labels = ["misinfo", "fact", "neutral"]

# claim subject
# this will be automated later
subject = "Hydroxychloroquine_full" + ".csv"

# dataset to factcheck
dataPath = os.path.join(os.path.abspath(".."), "test_data\\hcq_processed.csv")

# number of claims to check from dataset
sample_size = 10

class BertSemanticDataGenerator(tf.keras.utils.Sequence):
    """Generates batches of data.

    Args:
        sentence_pairs: Array of premise and hypothesis input sentences.
        labels: Array of labels.
        batch_size: Integer batch size.
        shuffle: boolean, whether to shuffle the data.
        include_targets: boolean, whether to incude the labels.

    Returns:
        Tuples `([input_ids, attention_mask, `token_type_ids], labels)`
        (or just `[input_ids, attention_mask, `token_type_ids]`
         if `include_targets=False`)
    """

    def __init__(
        self,
        sentence_pairs,
        labels,
        batch_size=batch_size,
        shuffle=True,
        include_targets=True,
    ):
        self.sentence_pairs = sentence_pairs
        self.labels = labels
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.include_targets = include_targets
        # Load our BERT Tokenizer to encode the text.
        # We will use base-base-uncased pretrained model.
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )
        self.indexes = np.arange(len(self.sentence_pairs))
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch.
        return len(self.sentence_pairs) // self.batch_size

    def __getitem__(self, idx):
        # Retrieves the batch of index.
        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        sentence_pairs = self.sentence_pairs[indexes]

        # With BERT tokenizer's batch_encode_plus batch of both the sentences are
        # encoded together and separated by [SEP] token.
        encoded = self.tokenizer.batch_encode_plus(
            sentence_pairs.tolist(),
            add_special_tokens=True,
            truncation = True,
            max_length=max_length,
            return_attention_mask=True,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_tensors="tf",
        )

        # Convert batch of encoded features to numpy array.
        input_ids = np.array(encoded["input_ids"], dtype="int32")
        attention_masks = np.array(encoded["attention_mask"], dtype="int32")
        token_type_ids = np.array(encoded["token_type_ids"], dtype="int32")

        # Set to true if data generator is used for training/validation.
        if self.include_targets:
            labels = np.array(self.labels[indexes], dtype="int32")
            return [input_ids, attention_masks, token_type_ids], labels
        else:
            return [input_ids, attention_masks, token_type_ids]

    def on_epoch_end(self):
        # Shuffle indexes after each epoch if shuffle is set to True.
        if self.shuffle:
            np.random.RandomState(42).shuffle(self.indexes)
            
            
def check_similarity(sentence1, sentence2):
    # had to hard code this part because the matcher was doing a bad job
    # 2 identical statements could go in and they would recieve a neutral verdict
    
    if (sentence1 == sentence2):
        return(1, 1)
    
    
    sentence_pairs = np.array([[str(sentence1), str(sentence2)]])
    test_data = BertSemanticDataGenerator(
        sentence_pairs, labels=None, batch_size=1, shuffle=False, include_targets=False,
    )

    proba = model.predict(test_data[0])[0]
    idx = np.argmax(proba)
    proba2 = proba[idx]
    return idx, proba2

def keyword_fact_check(claim, kb):
    # creating keywords from claim
    # removes all instances of "'s", all punctuation, all numbers, and sets to lowercase
    claim_keywords = claim.replace("’s",'').replace('[^\w\s]','').replace('[\d]','').lower()

    claim_keywords = claim_keywords.split(" ")
    # had a weird issue with fullstops
    claim_keywords = [word.replace(".", "") for word in claim_keywords]
    # removes stopwords
    claim_keywords = [word for word in claim_keywords if word not in (stop)]
    # removes duplicates
    claim_keywords = (" ").join([*set(claim_keywords)])
    
    # creating a new column, similarity, where value is vocab intersection of keywords divided by no. of claim keywords
    similarity_list = []
    
    for row in kb['keywords']:
        similarity = len(set(claim_keywords.split(" ")) & set(row.split(" ")))
        similarity_list.append(similarity)
        
    kb['similarity'] = similarity_list
    
    # the verdict decided during analysis
    current_verdict = 0
    # current highest certainty
    highest_similarity = 0
    # 
    best_evidence = ""
    
    # gets best fact based on keyword similarity
    # will sort contingency later (no obvious winner on keyword search)
    # either:
    # warning at low keyword match value
    # or gets other facts with same similarity (same as last approach)
    rows = kb[kb.similarity == kb.similarity.max()]
    
    # had to cut down number of facts taken in due to increasing runtime (122 facts for comparison on one claim leading to
    # >10 minutes of analysis, and one claim had 18000+ for comparison)
    if len(rows.index) > 20:
        rows = rows.sample(n = 20)
    
    #print("claim:", claim)
    
    # iterate for all those with max keywords
    for i in range(len(rows.index)):
        analysis = check_similarity(rows['statement'].iloc[i], claim)
        
        if analysis[1] > highest_similarity:
            highest_similarity = analysis[1]
            #print(rows['statement'].iloc[i], rows['verdict'].iloc[i])
            
            #best_evidence = rows['statement'].iloc[i]
            
            # inversion for non-facts in kb
            if not rows['verdict'].iloc[i]:
                match analysis[0]:
                    case 0:
                        current_verdict = 1 
                    case 1:
                        current_verdict = 0
                    case _:
                        current_verdict = analysis[0]
            else:
                current_verdict = analysis[0]
            
    #print("statement:", best_evidence)
    
    # just for testing
    if current_verdict == 2:
        current_verdict = 1
    
    return(current_verdict, highest_similarity)
    
model = tf.keras.models.load_model(saved_model_path)

# list of stopwords, from https://github.com/Alir3z4/stop-words/blob/bd8cc1434faeb3449735ed570a4a392ab5d35291/english.txt
# has been modified from this version

file = open("english.txt", "r")
stop = file.read()
file.close()

# get path to KB for subject
kbPath = os.path.join(os.path.abspath(""), "factbase")
kbPath = os.path.join(kbPath, subject)

kb = pd.read_csv(kbPath)

data = pd.read_csv(dataPath)

group_size = int(sample_size / 2)

# map then sample equally
data['pred'] = data['pred'].map({2:0, 1:1, 0:0})
data = data.groupby('pred', group_keys=False).apply(lambda x: x.sample(group_size))

transformers.logging.set_verbosity_error()

correct = 0
total = 0
# these are probably the real labels, but since I don't know, I'm not using them
#local_labels = ["neutral", "misinfo", "fact"]
#final_labels = ["misinfo", "fact", "neutral"]

local_labels = ["neutral", "misinfo"]
final_labels = ["misinfo", "neutral"]

start_time = timeit.default_timer()

for i in range(len(data.index)):
        analysis = keyword_fact_check(data['text'].iloc[i], kb)
        
        if analysis[0] == data['pred'].iloc[i]:
            correct += 1
        
        total += 1
        
        # print(final_labels[analysis[0]], local_labels[data['pred'].iloc[i]])

stop_time = timeit.default_timer()

output_path = os.path.abspath("")
file_name = output_path + "\\test_output.txt"
f = open(file_name, "x")
f.write(str(correct))
f.write("\n")
f.write(str(stop_time - start_time))
f.write("s")
f.close()