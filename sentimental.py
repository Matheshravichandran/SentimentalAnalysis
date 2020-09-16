# Import various libraries
# there are some libraries that are not downloaded on your system like the nltk.corpus and even if it is downloaded there are some inner libraries like positive tweets.json
# So I recommend you to install the libraries using the following command ----> download(library name)
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier

import re, string, random

# This function is used to remove the noise like the word a,the,an and punctuation marks and spaces and @-tags so that we could have a cleaner data

def remove_noise(tweet_tokens, stop_words = ()):

    cleaned_tokens = []
# Iterating over the tokenized list of tweets
    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
# lemmatizing means it gives the removed form of a word
# for example working becomes work by removing -ing form from the back of the word
        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)
# And by the way we change everything into the lowercase letters for hoaving a good model
        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens
# This function makes a clean tokenized list of words free from noise
def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token
#This function makes a dictionary of the items in the cleaned_tokens_list and assigns the second paramater of the dictionary as True
def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

if __name__ == "__main__":

    positive_tweets = twitter_samples.strings('positive_tweets.json')
    negative_tweets = twitter_samples.strings('negative_tweets.json')
    text = twitter_samples.strings('tweets.20150430-223406.json')
    tweet_tokens = twitter_samples.tokenized('positive_tweets.json')[0]
# downloading the stopwards like 'the', 'a', 'an' in the language english from stopwords library to variable stop_words
    stop_words = stopwords.words('english')
# assigning the words which are positive and negative from the respective json files to two different variables
    positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
    negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')
# Initializing empty lists for getting the words into them
    positive_cleaned_tokens_list = []
    negative_cleaned_tokens_list = []
# Now we are looping through the positive_tweet_tokens and we are adding them to the empty lists initialized above by removing the noise from the them such as stop words
    for tokens in positive_tweet_tokens:
        positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))
# Now we are looping through the negative_tweet_tokens and we are adding them to the empty lists initialized above by removing the noise from the them such as stop words
    for tokens in negative_tweet_tokens:
        negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))
 # Now we are getting all the words from the positive_cleaned_tokens_list into another variable all_pos_words
    all_pos_words = get_all_words(positive_cleaned_tokens_list)
# We can know how many times a word in the all_pos_words repeats itself
    freq_dist_pos = FreqDist(all_pos_words)
# Printing the frequency of the 10 most common words
    print(freq_dist_pos.most_common(10))
# Now we are getting the dictionary of words of positive_cleaned_tokens_list into a variable positive_tokens_for_model
    positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
# Now we are getting the dictionary of words of negative_cleaned_tokens_list into a variable positive_tokens_for_model
    negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)
#now we are looping through the above initialized arrays for getting positive defined and negative defined dictionaries from them.
    positive_dataset = [(tweet_dict, "Positive")
                         for tweet_dict in positive_tokens_for_model]

    negative_dataset = [(tweet_dict, "Negative")
                         for tweet_dict in negative_tokens_for_model]
# Now we are mixing the whole negative and positive dataset's and making a training model
    dataset = positive_dataset + negative_dataset
# We ares shuffling the data
    random.shuffle(dataset)
# We are dividing the dataset into two parts the first 7000 are defined as a training data set and the latter are defined as a testing dataset
    train_data = dataset[:7000]
    test_data = dataset[7000:]
# A classifier is used to run our dataset and classify it
    classifier = NaiveBayesClassifier.train(train_data)

    print("Accuracy is:", classify.accuracy(classifier, test_data))

    print(classifier.show_most_informative_features(10))

#my_array = ["I'm freaking happy", "I'm not happy with your bad attitude", "You are not as good as you look."]
#for custom_tweet in my_array:


# Please feel free to change the String in the custom_tweet variable as we can analyze different sentences using the above method
    custom_tweet = "It's not nice it's very bad."

    custom_tokens = remove_noise(word_tokenize(custom_tweet))


    print(custom_tweet)
    print(classifier.classify(dict([token, True] for token in custom_tokens)))