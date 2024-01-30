This is my masters' project for MSci Computer Science at Swansea University. It is a misinformation detection approach using a fact verification method. Implemented and tested using Python.

The idea behind the project was to use the BERT Large Language model to perform semantic matching between a knowledge-base of facts, and tweets.

The model was trained using the SNLI (Stanford Natural Language Inference) dataset.

The contents of the knowledge-bases was constructed from scraped wikipedia pages related to the topic.
The tweet data used was pre-labelled tweets relating to Coronavirus and fake cures and causes of it.

The solution also made use of a keyword search to find the most likely related data in the knowledge base to compare the tweet against.

Overall, the solution performed very poorly. There was a large mismatch between the model used, the data in the knowledge-base, and the actual tweet data. 
This made comparison between them difficult, and borderline pointless.

Given how noisy tweet data is, implementing any kind of automated fact-checking is very difficult.
A better solution might use a different model (one trained on tweet inference), different datasets (knowledge bases constructed from vetted tweets), or a different approach entirely.

The definitive version of the source code, along with the test scripts were lost on a trial Google Cloud instance. 
The source code provided here is the development version, before cloud deployment.
