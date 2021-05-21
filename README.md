# Hotel-Recommendation-with-Recommender-Systems

Comparative Evaluation and Discussion
5.1 Neural Network Embedding Model
In this stage of the study, 30-dimensional hotel embedding vectors are obtained successfully. The TSNE and UMAP algorithms map high-dimensional vectors two dimensional vectors to visualize the vectors in 2D system [44, 45]. The 2D visualization of hotel embeddings with TSNE algorithm is given in Figure 5.1.
![image](https://user-images.githubusercontent.com/24464017/119191430-4c657180-ba87-11eb-93c6-020b2b6a607a.png)
Figure 5.1: Hotel embedding vectors visualization
According to UMAP representation, hotels can be grouped under two clusters. This is also logical assumption since some of the hotels are convenience or close to convenience hotel model as hostels or low budget hotels just offer breakfast and room, others are high segment such as offering three meals in a day or pool or specialized services with high prices. UMAP representation is much faster than TSNE representation in means of time.
As mentioned in the third chapter, also similarity of the hotels is calculated regarding the cosine similarity. Similarity result for hotel id 5101 is given the Figure 5.2. As seen in the Figure, the hotels with less cosine distance to 5101 with other words most similar hotels to selected hotel is also closest hotels in the TSNE map.
30
Figure 5.1: TSNE representation of similar hotels to 5101
It is also possible to obtain similarity scores from the model by sorting and take reverse of the cosine distances vector for n objects. The similarity scores of hotel id 5101 is given in Figure 5.2 where n = 10.
Figure 5.2: Top 10 similar hotels for hotel id 5101
From the cosine distances, it is also possible to obtain the least similar hotels. The least similar 10 hotel for hotel id 5101 is given in Figure 5.3.
31
Figure 5.3: The least similar 10 hotels for hotel id 5101
The most similar and the least similar hotels also can be seen in same plot as in Figure 5.4.
Figure 5.4: The least and most similar hotels for hotel id 5101
In the literature of ACM RecSys’19, the Word2Vec embedding model was popular since the dataset also had sequence information. The DeepWalk algorithm for the hotel recommendation problem is firstly introduced in this study. A similar example work is a Hotel2Vec model as mentioned at the end of the second chapter. The Hotel2Vec model also implements a hotel embedding vector from amenities, however, it also includes the geography of the hotel to embedding producing phase. In this graduation thesis, the city feature was used in prediction filtering and meta input for RNN models.
32
Two different models are tested based on pure embeddings. The first one takes the last impression item and returns the distances of the cosine distances in a sorted manner. After filtering the hotels in the same city with the last impression, the closest hotel is chosen as the prediction. This approach achieved a 0.1473 MRR score in the 20.000 sample dataset and a 0.1371 MRR score in the 50.000 dataset. This score means when the engine recommends a ~7 hotel to the user, the target hotel will be in it.
Another one calculates the sequential average and finds the closest hotel embedding vector to the average vector. This method has applied on the same dataset as the last impression item method and scored 0.10149 in the 20.000 dataset and 0.118582 in the 50.000 dataset. This result means if the engine recommends ~10 hotels to the user, the recommendations will include the target hotel.
When comparing the results of the two models, the last impression item has superior performance than the sequential average. From that information, it can be inducted that the last clicks should have more weight regarding the previous ones. This is also the same result with the attention map of the Multi Input One Output Bi-RNN model.
5.2 Multi Input One Output Bi-RNN
The Multi Input One Output Bi-RNN model is implemented with previously calculated embedding vectors. First, it has been tested without filtering predictions regarding the city factor. The MRR score in a 20.000 dataset with 15 epoch and 16 batch size is 0.00050149. Since RNN models need more data to learn better, the approach has been tested on the bigger dataset. The MRR score in the 100.000 dataset with 18 epochs and 16 batch size is 0.00040919. So, it is even worse than the smaller dataset.
They are very low scores when comparing with the very simple models as described in earlier subsections. From here, it is easily understood that without giving enough importance to the city feature it is not possible to obtain a good score since when the city factor is discarded the engine should give a prediction on 927142 hotels. Despite that, if the city factor is counted as a filter, then the highest amount of hotel reduces to 2500 for a city.
The models are tested once again with the same setup plus a city filter. The MRR scores are dramatically increased. For the 20.000 dataset, it increases from 0.00050149 to 0.11328. It also increased from 0.00040919 to 0.109979 for the 50.000 dataset.
Final scores are better than the sequential average model but worse than the last impression item based predictions. It indicates the same result with the previous subsection that is the last item is more important than the whole sequence.
Even though the increase of the scores is more than double, these scores are not good enough to outperform the leaderboard of the RecSys or to meet the goals in the interim report.
33
5.3 Multi Input One Output Bi-RNN with Attention
The previously implemented model has been expanded with attention mechanism on top of the Bi-directional layer. It made faster the training part. The model has been tried on different datasets with different epoch and batch sizes as in Table 5.1.
Table 5.1: The result of the Multi Input One Output Bi-RNN with Attention
Dataset
Epoch
Batch Size
MRR Score
Top K (k=100) Categorical Accuracy 50.000 3 16 0.10347 0.5019
100.000
10
16
0.10943
0.3412 1.000.000 15 8 0.11123 0.4154
As seen in the above, the results are very similar to without attention model. The only change is training phase is faster now. However, if the model can be trained more and hyperparameter tuning can be applied, the results may change since the loss is steadily decreasing during the training.
Also, if an attention heatmap can be extracted from the attention weights, it may give an idea why the model is not performing better than other models. So, even the performance could not be superior than previous one still this model is useful to learn how to improve the architecture.
Another reason could be that the lack of encoder-decoder structure in the architecture. Since recently attention mechanism is seen as an improved part of autoencoder models, adding encoder-decoder layers could improve the result.
These results are also not good enough to meet the interim report criterions or to outperform the proposed models. However, it can be found promising with the improvement probabilities and the information that the model can provide.
