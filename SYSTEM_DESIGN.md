# System Design Document for Anime Recommendation System

## Overview
The Anime Recommendation System is designed to provide personalized anime recommendations based on user ratings. It utilizes a Flask web application to handle requests and responses and employs Non-negative Matrix Factorization (NMF) for generating recommendations.

## Architecture
The system is structured as follows:

1. **Flask Web Framework**: The application is built using Flask, which allows for easy handling of HTTP requests and responses. The main routes include:
   - `/`: A welcome endpoint.
   - `/anime`: Queries the anime dataset based on various filters. E.g. `/anime?mt_episodes=200&lt_episodes=300&mt_rating=7&mt_members=1000&genre=Drama` queries for a Drama anime with 200 to 300 episodes, a rating of more than 7, and more than 1000 members.
   - `/user`: Queries the user dataset based on various filters. E.g. `/user?user_id=567&mt_rating=5&lt_rating=7` queries for all entries of user 567 with a rating of more than 5 and less than 7.
   - `/recommend/<user_id>`: Provides recommendations for a specified user. The user can specify the number of recommendations to return using the `k` parameter. If not specified, the default value is 5.
   - `/talk/<user_id>/<user_message>`: Provides a more natural experience for the user to get recommendations.

2. **Data Processing**:
   - **Preprocessing**: The `preprocess_data` function cleans and prepares the data by handling missing values and transforming data types. E.g. `episodes` makes sense being an int, while `genre` and `type` are more like strings. `NaN` ratings are replaced with -1 because they watched the anime but it has no rating. If there are duplicates, I keep the last which is probably the most recent.
   - **Feature Engineering**: The `feature_engineer` function extracts relevant features from the ratings and anime data, which can be used for improving the recommendation algorithm. They are intended to express the following ideas:
     - Users have different scales of ratings, so the average can be useful for normalizing the ratings.
     - If a user watched a lot of anime of the same kind, even if they didn't rate it, it's likely that they enjoy it. So watch counts are also useful as ratings.
     - Some users like an anime to have plenty of episodes, so they can continue watching it. Others prefer shorter anime so they don't get stuck binge-watching it.
     - Some people prefer popular anime, while others enjoy the thrill of discovering new unknown anime.
     - Some people prefer TV over movie aesthetics or vice versa.
     - Every one of these features can be used to generate recommendations and the best model will probably use a combination of them, like BellKor Solution to the Netflix Grand Prize[^1].
     
3. **Recommendation Algorithm**: 
   - **Model Selection**: According to Netflix[^2], there's no single best algorithm for recommender systems and they add and remove new models working in parallel, using an ensemble of the best models, which makes sense since *bagging* and *boosting* are the most commonly used techniques for improving the performance of models. However, according to Stephen Gower[^3], the winner of the Netflix Prize used mainly an iterative version of SVD combined with Restricted Boltzmann Machine (RBM) algorithm. I decided to use this line of work, but in a simpler way for the purpose of this exercise using mainly Non-negative Matrix Factorization (NMF).
   - **NMF**: The Non-negative Matrix Factorization is a technique similar to SVD, but it uses an approximation method to decompose a matrix into non-negative sparse `W` and `H` matrices. Being bounded by 0 makes the model more interpretable and allows for easier interpretation of the results. The `nmf()` function implements Non-negative Matrix Factorization to decompose the user-item matrix into latent factors, which we can interpret as clusters of similarity. This technique is very popular and largely employed in topic modeling, dimensionality reduction, diarization, and others besides recommender systems.
   - **Paremeters**: The *NMF* only needs one parameter `n_components`, which is size of its latent dimensions. I chose `n_components=20` because it also relates to the anime groups they are related and 20 is close to half of the number of genres, so it can superpose more than one genre like "School" and "Drama" in the same group. This parameter makes 95% of the groups lie within 2 standard deviations of the mean of group users, implying that they are distributed similarly, which is a good sign. If the groups had very different numbers of users (some with too many users, and others with almost none), it would mean the model wasn't very capable of capturing the latent differences between them. 
   - **Recommendation Logic**: I implemented the simplest version of this logic, where I assume the column where the biggest value of the row is the user's group. However, a more nuanced approach is to assume the user is similar to each group according to the weight of each column. This remains to be implemented and tested. After deciding the user group, the animes are ranked according to the weights of the group in matrix `H`.
   - **Next Steps**: The model can be improved by using an ensemble method, similar to Netflix, which is a technique that combines multiple models to improve the performance of the final model. The final weight of the recommendation will be `r_f = w1 * r1 + w2 * r2 + ... + wn * rn`, where `r_f` is the final recommendation, `r1` to `rn` are the recommendations from the individual models, and `w1` to `wn` are the weights of each model. This is intuitive as we can decompose our likings into factors and then combine them to get a final score, e.g. I would like to get Cowboy Bebop recommended to me because everybody is talking about it and I don't want to miss out, I like a good soundtrack, I like fighting and sci-fi, and there aren't so many episodes like One Piece, which I will probably never watch. On an even more distant improvement, the model could consider that the users' taste changes over time, so the models could be adapted to the user's tastes.

4. **Data Storage**:
   - **Pickle Files**: Intermediate results, such as NMF components and processed data, are stored in pickle files for efficient retrieval and reuse, minimizing the need for repeated computations. It could be improved the size and speed using parquets and/or databases but I wanted to keep it simple for now.

5. **Evaluation**:
   - **Challenge**: Since the data is sparse, i.e. users watched and rated only a tiny subset of available anime, traditional "plug and play" methods of evaluation are not applicable because it's hard for the system to recommend something already watched in the test set. Taking this challenge into account, I decided that I would get a particular subset of data: only rated anime with users with more than 20 ratings and making sure that the anime sampled is from the top and bottom quantiles of ratings for each user so that it's possible to cross-validate the model with the data. 

6. **LLMs**:
   - **GPT4o**: Due to time constraints, I decided to use GPT4o for the final recommendation for it's a very well-ranked model and I already have more experience with it.
   - **LLM Recommendation Logic**: The GPT4o model gets a list of fixed top 10 anime recommendations based on the user's group and uses it as context for the user prompt and then picks one of the recommendations based on the user's prompt without any memory of past conversations.
   - **Next Steps**: This is a very simple approach that can be improved by using tools. The simplest way to use them is the Langchain Dataframe agent. However, since it's just a simple wrapper on top of a Python agent that can execute arbitrary Python code, I decided not to implement it without time to also implement guardrails to avoid malicious code execution. A more thoughtful approach would be to implement a llamaindex agent workflow. This could be done by orchestrating agents for different tasks based on more common recommendation requests and even having one Python agent for a edge cases guarded by guardrails.
   - **Local LLM**: When choosing to transition to a LLM deployed in-house, the first step is to choose and evaluate well the open source models. Today there are competitive instruct-tuned foundation models like Llama3.1 405B and smaller models like WizardLM-2 8x22B and Llama3.2 70B. The model could be deployed in a servers like AWS, Azure or Lightning AI, with API endpoints for recommendation requests. 

## Technologies Used
- **Python**: The primary programming language for the application.
- **Flask**: The web framework used to create the API.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Scikit-learn**: For implementing the NMF algorithm.
- **Docker**: For containerization, ensuring consistent deployment across environments.
- **OPENAI**: For the LLM API.

---

## Next steps:

- [ ] When no parameter is provided for the `/user` and `/anime` endpoints, use it as an opportunity to teach its commands.
- [ ] Improve the ML recommendation model using bagging.
- [ ] Improve LLM flexibility of recommendation by using an agent workflow for less obvious recommendation requests.
- [ ] Use a Local LLM for recommendation requests.
- [ ] Implement an evaluation framework for the recommendation model.

[^1]: [The BellKor Solution to the Netflix Grand Prize](https://www2.seas.gwu.edu/~simhaweb/champalg/cf/papers/KorenBellKor2009.pdf)

[^2]: [Netflix Recommendations: Beyond the 5 stars (Part 2)](https://netflixtechblog.com/netflix-recommendations-beyond-the-5-stars-part-2-d9b96aa399f5)

[^3]: Stephen Gower. [Netflix Prize and SVD](http://buzzard.ups.edu/courses/2014spring/420projects/math420-UPS-spring-2014-gower-netflix-SVD.pdf). April 18th 2014 