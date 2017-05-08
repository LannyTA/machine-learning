#fire up graphlab
import graphlab
products=graphlab.SFrame('amazon_baby.gl/')
#build a word_count vector
products['word_count']=graphlab.text_analytics.count_words(products['review'])
#examing the most sold product
giraffe_review=products[products['name']=='Vulli Sophie the Giraffe Teether']
#define positive and negative
products=products[products['rating']!=3]
products['sentiment']=products['rating']>=4
#build a classifier
train_data,test_data=products.random_split(.8,seed=0)
sentiment_model=graphlab.logistic_classifier.create(train_data,
                                                    target='sentiment',
                                                    features=['word_count'],
                                                    validation_set=test_data)
#applying this model
giraffe_review=products[products['name']=='Vulli Sophie the Giraffe Teether']
giraffe_review['predicted_sentiment']=sentiment_model.predict(giraffe_review,output_type='probability')
#show some result
giraffe_review=giraffe_review.sort('predicted_sentiment',ascending=False)
print(giraffe_review[0]['review'])