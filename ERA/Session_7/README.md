**Step 1:**

**Targets:** Reduce the number of parameters to under 10k (Reduced the number of epochs to 15)

**Results:**

Best train accuracy: 98.66

Best test accuracy: 99.31

Total parameters: 7752
  
**Analysis:** Reducing the total number of parameters by changing the model. As expected, it reduced the accuracy but not my much.

**File Link:** https://github.com/paishowstopper/TSAI/blob/main/ERA/Session_7/ERA_S7_1.ipynb



**Step 2:**

**Targets:**  Test accuracy should be above 99.4 (at least once)

**Results:** 

Best train accuracy: 99.37

Best test accuracy: 99.42

Total parameters: 7752
  
**Analysis:** Removed dropout to get accuracy values close to our final target. Crossed 99.4 twice but not consistently. Before removing dropout, tried to change its values. Changed to between 0.5 to 0.8 for middle layers which didn't show any improvement. Increasing it to 0.25 degraded the performance.

**File Link:** https://github.com/paishowstopper/TSAI/blob/main/ERA/Session_7/ERA_S7_2.ipynb



**Step 3:**

**Targets:** Test accuracy should be above 99.4 consistently

**Results:** 

Best train accuracy: 99.39

Best test accuracy: 99.46

Total parameters: 7752

**Analysis:** Changed the step size of the LR scheduler to 8. The accuracy values were consistent now.

**File Link:** https://github.com/paishowstopper/TSAI/blob/main/ERA/Session_7/ERA_S7_3.ipynb
