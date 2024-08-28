# System Design Document for Anime Recommendation System

## Overview
The Anime Recommendation System is designed to provide personalized anime recommendations based on user ratings. It utilizes a Flask web application to handle requests and responses, and employs Non-negative Matrix Factorization (NMF) for generating recommendations.

## Architecture
The system is structured as follows:

1. **Flask Web Framework**: The application is built using Flask, which allows for easy handling of HTTP requests and responses. The main routes include:
   - `/`: A welcome endpoint.
   - `/upload`: Accepts CSV files containing user ratings and anime data.
   - `/recommend/<user_id>`: Provides recommendations for a specified user.

2. **Data Processing**:
   - **Preprocessing**: The `preprocess_data` function cleans and prepares the data by handling missing values and transforming data types.
   - **Feature Engineering**: The `feature_engineer` function extracts relevant features from the ratings and anime data, which are essential for the recommendation algorithm.

3. **Recommendation Algorithm**:
   - **NMF**: The `nmf` function implements Non-negative Matrix Factorization to decompose the user-item matrix into latent factors. This allows the system to identify patterns in user preferences and generate recommendations.
   - **Recommendation Logic**: The `recommend` function uses the decomposed matrices to suggest anime that a user has not yet watched, based on their preferences.

4. **Data Storage**:
   - **Pickle Files**: Intermediate results, such as NMF components and processed data, are stored in pickle files for efficient retrieval and reuse, minimizing the need for repeated computations.

5. **Concurrency**:
   - **Threading**: The application uses Python's threading module to handle background tasks, allowing the main application to remain responsive while performing long-running operations.

## Technologies Used
- **Python**: The primary programming language for the application.
- **Flask**: The web framework used to create the API.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Scikit-learn**: For implementing the NMF algorithm.
- **Docker**: For containerization, ensuring consistent deployment across environments.

## Conclusion
This architecture provides a scalable and efficient way to deliver personalized anime recommendations. The use of NMF allows for effective pattern recognition in user preferences, while Flask and Docker facilitate easy deployment and interaction with the application.