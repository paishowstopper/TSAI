1. Upload the Validation Accuracy Change Graph (all 5 models combined) - 100 pts
2. Upload the Loss Change Graph (all 5 models combined) - 100 pts

![image](https://user-images.githubusercontent.com/26896746/118090455-38ff3a00-b3e7-11eb-826f-10559247cb0e.png)

3. Upload the image showing 25 misclassified images for the "with GBN" model. - 250 pts

![image](https://user-images.githubusercontent.com/26896746/118090593-61873400-b3e7-11eb-95a8-69abc95a9164.png)

4. Explain your observation w.r.t. L1 and L2's performance in the regularization of your model. - 50 pts

GBN and L1 + BN are the best performing models overall. Both start with a relatively higher validation accuracy and lower test loss than the other models. W.r.t test loss, both finish training at a much lower loss compared to the other 3 (GBN slightly better than L1+BN). W.r.t validation accuracy, even though these 2 show higher accuracy at the beginning, by the end of the training, all 5 models are quite close to each other. 
