**Step 1:**

**Targets:** Reduce the number of parameters to under 10k (Reduced the number of epochs to 15)

**Results:**

Best test accuracy: 99.36

Final test accuracy: 99.26

Total parameters: 7752
  
**Analysis:** Reducing the total number of parameters by changing the model. As expected, it reduced the accuracy but not my much.

**File Link:** https://github.com/paishowstopper/TSAI/blob/main/EVA5/Session_5/EVA5_S5_1.ipynb

**Step 2:**

**Targets:** Test accuracy should be consistent (Should not drop well below best accuracy)

**Results:** 

Best test accuracy: 99.37

Final test accuracy: 99.35

Total parameters: 7752
  
**Analysis:** Changed the step size of the LR scheduler to 7 (Step sizes 6 and 8 were degrading the accuracy). The accuracy values were consistent now.

**File Link:** https://github.com/paishowstopper/TSAI/blob/main/EVA5/Session_5/EVA5_S5_2.ipynb

**Step 3:**

**Targets:** Test accuracy should be above 99.4 (at least once)

**Results:** 

Best test accuracy: 99.43

Final test accuracy: 99.37

Total parameters: 7752

**Analysis:** Removed dropout to get accuracy values close to our final target. Crossed 99.4 twice but not consistently. Before removing dropout, tried to change its values. Changed to between 0.5 to 0.8 for middle layers which didn't show any improvement. Increasing it to 0.25 degraded the performance.

**File Link:** https://github.com/paishowstopper/TSAI/blob/main/EVA5/Session_5/EVA5_S5_3.ipynb

**Step 4:**

**Targets:** Test accuracy should be above 99.4 consistently

**Results: **

Best test accuracy: 99.49

Final test accuracy: 99.47

Total parameters: 7752

**Analysis:** Removed the padding of convolution block 7 and the model started performing very well from epoch 8 onwards. Consistently, hitting above 99.4 till the end.

**File Link:** https://github.com/paishowstopper/TSAI/blob/main/EVA5/Session_5/EVA5_S5_4.ipynb
