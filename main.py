import nltk.classify.util
nltk.download()
from nltk.classify import NaiveBayesClassifier

def extract_features(word_list):
  return dict([(word, True) for word in word_list])

//extract positive and neg tags
positive_fileids = movie_reviews.fileids('pos')
negative_fileids = movie_reviews.fileids('neg')

fp= [(extract_features(movie_reviews.words(fileids=[f])), 'Positive') for f in positive_fileids]
fn = [(extract_features(movie_reviews.words(fileids=[f])),'Negative') for f in negative_fileids]

//data spliting
TF = 0.85
TS = int(TF * len(fp))
TN = int(TF * len(fn))
features_train= fp[:TS] + fn[:TN]
features_test =fp[TS:]+ fn[TN:]

//Fit data in the classifier
classifier = NaiveBayesClassifier.train(features_train)
print("\nAccuracy : ", nltk.classify.util.accuracy(classifier, features_test)*100)

//Take Data from User
review =input();

print ("\nPredictions:")
probdist = classifier.prob_classify(extract_features(review.split()))
pred_sentiment = probdist.max()
        
print ("Predicted sentiment:", pred_sentiment )
print ("Probability:", round(probdist.prob(pred_sentiment), 2))
