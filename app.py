from flask import Flask, render_template, request
import pandas as pd
import gdown
import os
import ast
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Google Drive file download link for the main dataset
drive_id = "1RHu6Tk3vW63nMau14S7v6onRxkAk_x9A"
output_csv = "ecommerce_data_new.csv"

# Check if the file already exists locally, otherwise download it
if not os.path.exists(output_csv):
    gdown.download(f"https://drive.google.com/uc?export=download&id={drive_id}", output_csv, quiet=False)

# Load the datasets
data = pd.read_csv(output_csv)

# Remove duplicate products, keeping the first occurrence by parent_asin
data_unique = data.drop_duplicates(subset='parent_asin', keep='first')

# Function to get the hi_res image link
def get_hi_res_image(images):
    try:
        images_list = ast.literal_eval(images)
        if images_list and 'large' in images_list[0]:
            return images_list[0]['large']
        else:
            return 'static/images/image-not-found.jpg'
    except Exception as e:
        return 'static/images/image-not-found.jpg'

# Apply the function to the images column
data_unique['hi_res_image'] = data_unique['images'].apply(get_hi_res_image)

# Load the preprocessed user-item matrix
user_item_matrix = pd.read_csv('user_item_matrix.csv', index_col='user_id')
original_nan_mask = pd.read_csv('original_nan_mask.csv', index_col='user_id')

# Compute user-user similarity matrix using cosine similarity
user_similarity = cosine_similarity(user_item_matrix)


# Function to get the top seasonatl products for the current month
def get_top_seasonal_products():
    #Convert timestamp column to datetime objects
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['month'] = data['timestamp'].dt.month
    
    #Get current month
    current_month = datetime.now().month
    
    #Group by asin ,month and count purchases
    purchase_count = data.groupby(['asin', 'month']).size().reset_index(name='purchase_count')
    
    #Group by asin, month and calculate average ratings
    average_rating = data.groupby(['asin', 'month'])['rating'].mean().reset_index(name='average_rating')
    
    #Merge purchase counts and average ratings
    monthly_item_popularity = pd.merge(purchase_count, average_rating, on=['asin', 'month'])
    
    #Sort by popularity and average rating
    popular_items = monthly_item_popularity[monthly_item_popularity['month'] == current_month]
    popular_items = popular_items.sort_values(by=['purchase_count', 'average_rating'], ascending=[False, False])
    
    #Get top 5 ASINs
    top_5_asins = popular_items['asin'][:5].tolist()
    
    #Fetch the product details
    top_5_products = data_unique[data_unique['asin'].isin(top_5_asins)][['asin', 'product_title', 'hi_res_image', 'average_rating']]
    
    return top_5_products.to_dict(orient='records')

@app.route('/')
def index():
    # Pagination logic for main products
    page = request.args.get('page', 1, type=int)  # Get the current page number
    per_page = 20  # Number of products per page
    paginated_data = data_unique.iloc[(page - 1) * per_page: page * per_page]

    # Get the top 5 seasonal products
    top_5_products = get_top_seasonal_products()

    # Render the homepage with both paginated products and hot products
    return render_template('index.html', results=paginated_data.to_dict(orient='records'), 
                           top_5_products=top_5_products, page=page)

@app.route('/search', methods=['POST', 'GET'])
def search():
    # Check if the request is a POST or GET to retrieve the query
    if request.method == 'POST':
        query = request.form['query']
    else:
        query = request.args.get('query', '')  # Get the query from URL parameters if navigating through pages

    # Filter results by product title or main category
    results = data_unique[
        (data_unique['product_title'].str.contains(query, case=False, na=False)) |
        (data_unique['main_category'].str.contains(query, case=False, na=False))
    ]
    
    # Pagination logic
    page = request.args.get('page', 1, type=int)  # Get the current page number
    per_page = 20  # Items per page
    paginated_results = results.iloc[(page - 1) * per_page: page * per_page]

    # get seasonal products
    top_5_products = get_top_seasonal_products()

    # Render the search results with pagination
    return render_template('index.html', query=query, results=paginated_results.to_dict(orient='records'), 
                           top_5_products=top_5_products, page=page)

@app.route('/product/<asin>')
def product_detail(asin):
    # Filter the product by asin for detailed view
    product = data_unique[data_unique['asin'] == asin].iloc[0]
    return render_template('product_detail.html', product=product)


@app.route('/collab-recommend/<user_id>')
def collab_recommend(user_id):
    # Check if the user exists in the user_item_matrix
    if user_id not in user_item_matrix.index:
        return f"No data available for user {user_id}", 404
    
    # Use original_nan_mask to find unrated items (originally NaN)
    unrated_items = original_nan_mask.loc[user_id]
    unrated_items = unrated_items[unrated_items].index  # Filter the originally unrated items
    
    if len(unrated_items) == 0:
        return f"User {user_id} has rated all items.", 200
    
    # Compute similarity scores between the target user and other users
    similarity_scores = user_similarity[user_item_matrix.index.get_loc(user_id)]

    # Sort users by similarity scores (excluding the user themselves)
    similar_users_indices = similarity_scores.argsort()[::-1][1:]
    
    # Get the ratings of similar users for the unrated items
    similar_users_ratings = user_item_matrix.iloc[similar_users_indices][unrated_items]

    # Calculate predicted ratings for the unrated items based on similar users' ratings
    weighted_ratings = similar_users_ratings.T.dot(similarity_scores[similar_users_indices])
    similarity_sums = similarity_scores[similar_users_indices].sum()

    predicted_ratings = weighted_ratings / similarity_sums
    
    # Get the top N recommended items (let's say top 5)
    top_recommended_items = predicted_ratings.nlargest(5).index
    
    # Retrieve product details from the original dataset for these recommendations
    recommended_products = data[data['asin'].isin(top_recommended_items)]

    # If no products found, return a message
    if recommended_products.empty:
        return f"No recommendations available for user {user_id}", 200
    
    # Render the recommendations page with the recommended products
    return render_template('recommendations.html', products=recommended_products.to_dict(orient='records'), user_id=user_id)

if __name__ == '__main__':
    app.run(debug=True)
