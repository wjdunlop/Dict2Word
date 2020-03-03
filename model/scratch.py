from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
stop_words = stopwords.words('english')
print(stop_words)

if "aren't" in stop_words:
	print("YEs")
negations = ["no", "not"]
for h in negations:
	stop_words.remove(h)
print(stop_words)