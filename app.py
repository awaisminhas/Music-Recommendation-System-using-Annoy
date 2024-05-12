from pyspark.sql import SparkSession

# Create Spark session
spark = SparkSession.builder.appName('fma_recommendation_system').getOrCreate()

from pymongo import MongoClient

# Set up MongoDB connection
client = MongoClient("mongodb://localhost:27017")

db = client['mfcc_database']
collection = db['mfcc_collection']

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType, FloatType

# Define the schema for the dataframe
schema = StructType([
    StructField('_id', StringType(), True),
    StructField('artist_name', StringType(), True),
    StructField('tags', ArrayType(StringType()), True),  # Changed tags to ArrayType(StringType())
    StructField('genre', StringType(), True),
    StructField('plays', IntegerType(), True),
    StructField('title', StringType(), True),
    StructField('mfcc_features', ArrayType(FloatType(), True))
])

# Get data from collection, limited to 1000 documents
data = collection.find().limit(5000)

# Convert data into a Spark dataframe using the defined schema
df = spark.createDataFrame(list(data), schema=schema)

# Drop rows with missing values in necessary columns
df = df.dropna(subset=["_id", "artist_name", "tags","genre","plays","title","mfcc_features"])

from annoy import AnnoyIndex

# Initialize Annoy index
num_features = len(df.first()['mfcc_features'])
annoy_index = AnnoyIndex(num_features, 'angular')  # 'angular' distance works well with cosine similarity


# Add items to Annoy index
for i, row in enumerate(df.collect()):
    audio_features = row['mfcc_features']
    annoy_index.add_item(i, audio_features)

# Build Annoy index

annoy_index.build(50)  # 50 trees for the index (adjust as needed)

def find_similar_items(audio_features, n=10):
    similar_items = annoy_index.get_nns_by_vector(audio_features, n)
    return [df.collect()[idx] for idx in similar_items]
    
    
from flask import Flask, render_template, request, jsonify
import random

app = Flask(__name__)

# Function to show 10 random music IDs and titles
def show_random_music():
    random_selection = random.sample(df.collect(), 10)
    return random_selection

# Function to get nearest recommendations for a given music ID
def get_nearest_recommendations(music_id):
    # Find similar items for the input music ID
    audio_features = df.filter(df._id == music_id).select("mfcc_features").collect()[0][0]
    similar_items = find_similar_items(audio_features)
    return similar_items

@app.route('/')
def index():
    random_music = show_random_music()
    return render_template('index.html', random_music=random_music)

@app.route('/recommendations', methods=['POST'])
def recommendations():
    music_id = request.form['music_id']
    nearest_recommendations = get_nearest_recommendations(music_id)

    recommendations = [(item[0], item[5]) for item in nearest_recommendations]

    return jsonify(recommendations)


if __name__ == '__main__':
    app.run(debug=True)


