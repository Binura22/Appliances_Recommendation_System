from flask import Flask, render_template, request
import pandas as pd
import gdown
import os
import ast

app = Flask(__name__)

# Google Drive file download link
drive_id = "1RHu6Tk3vW63nMau14S7v6onRxkAk_x9A"
output_csv = "ecommerce_data_new.csv"

# Check if the file already exists locally, otherwise download it
if not os.path.exists(output_csv):
    gdown.download(f"https://drive.google.com/uc?export=download&id={drive_id}", output_csv, quiet=False)

# Load the dataset
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

@app.route('/')
def index():
    # Pagination logic
    page = request.args.get('page', 1, type=int)  # Get the current page number
    per_page = 20  # Number of products per page
    paginated_data = data_unique.iloc[(page - 1) * per_page: page * per_page]

    return render_template('index.html', results=paginated_data.to_dict(orient='records'), page=page)

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

    # Render the search results with pagination
    return render_template('index.html', query=query, results=paginated_results.to_dict(orient='records'), page=page)

@app.route('/product/<asin>')
def product_detail(asin):
    # Filter the product by asin for detailed view
    product = data_unique[data_unique['asin'] == asin].iloc[0]
    return render_template('product_detail.html', product=product)

if __name__ == '__main__':
    app.run(debug=True)
