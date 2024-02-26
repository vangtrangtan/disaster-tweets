# disaster-tweets

Preprocess data:
  - nltk stemmer (also tried lemmatize but accuracy become worse)
  - mapping/normalize http link  in text
  - drop duplicate text

ML:

  - use Log Regression, Word Embeddings (count vector) with accuracy 79%
  - then use KNN to group similar texts on testdata, check manually to hardcode prediction, this increases 1% accuracy

Finally got 80.6% accuracy, rank 300/800

What's next ?

When I explore training data, I seek out one thing that we can split data into two parts. Good data contains 5064/7613 tweets, this set data is good since we can predict with 96% accuracy. Bad data contains 2549/7613 tweets with 58% accuracy. So how to take advantage this to get more accuracy on test data ?
