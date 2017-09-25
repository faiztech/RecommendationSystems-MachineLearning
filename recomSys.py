#A Hybrid Recommendation System which recommends movies
#Mohammed Faizuddin

#importing dependencies
import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

#creating a dataset by fetching using the imported 'fetch_movielens()'
data = fetch_movielens(min_rating=4.0)

#printing training and testing data
#using 'repr()' to return a string representating printable version of object
print(repr(data['train']))
print(repr(data['test']))

#creating a model using LightFM class and passing the parameter 'loss'
#this calculates the loss between our model's prediction and desired output
#WARP - Weighted Approximate - Rank Pairwise
model = LightFM(loss='warp')

#training the model using 'fit()'
#num_threads = num of parallel computations
model.fit(data['train'],epochs=30,num_threads=2)

#creating a recommendation function
def recommendations(model,data,user_ids):
    #storing num of users and num of items using the '.shape' attribute
    n_users, n_items = data['train'].shape

    #iterating through each user id and checking for positives and storing in csr format
    for user_id in user_ids:
        #storing the movies they already like
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

        #predicting movies they might like using 'predict()'
        scores = model.predict(user_id,np.arange(n_items))

        #just arranging them in order from most liked to least liked
        top_items = data['item_labels'][np.argsort(-scores)]

        #printing out recommendations
        print("\n***************")
        print("User %s" % user_id)

        print("     What you like:")
        for x in known_positives[:3]:
            print("         %s" %x)

        print("\n     What you may like:")
        for x in top_items[:3]:
            print("         %s" %x)
#end of recommendations function

#calling the function
recommendations(model, data, [5,15,400])
