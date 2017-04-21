## Notes
- With a static set of GVFS, we can predict the velocity for any spot that there's an associated GVF for that encoder position
- Predicted bit 7 never predicts anything. I suspect that's because the next state for that position is slightly unpredictable. It sometimes is 7. It sometimes is something else. About 50/50. So will never get over the threshold
- Bits 8,9, 19 are never observed so never contain information
- I changed the experiment up to "cull" the worst performing GVFs after 14000 observations have been made. Repeating that 30 times ends up with all the correct GVFs
- Looking at determinng when to cull in a more intelligent way. So looking at RUPEE
- Measured RUPEE for predicting "good" bits as well as "noisy" bits.
--- RUPEE for good bits went quickly to0.13. Then gradually went down to 0.008 after several thousad
--- RUPEE for noisy bits went to 0.13 quickly as well. But only went down to about 0.06
- Created AverageErrorKullEvery5000.png showing the error reduce over 100 runs of 90000 timesteps.
--- Killing 4 after 5000. Killing 1 every 5000 afterwards

## TODO
- use Rupee to determine when to kill off GVFs
- Experiment using GVFS crafting random features rather than feeding into higher level predictor
- Use Radial basis to choose next GVF bit