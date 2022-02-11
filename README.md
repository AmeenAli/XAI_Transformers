# Official Code Implementation of The Paper : XAI for Transformers: Better Explanations through Conservative Propagation
<p align="center">
  <img width="1100" height="300" src="https://i.ibb.co/XJ1wWyP/Screen-Shot-2022-02-11-at-11-23-58.png">
  <br>
  <img width="600" height="300" src="https://i.ibb.co/QdbXFjY/Screen-Shot-2022-02-11-at-11-26-06.png">
</p>

For the SST-2 and IMDB expermints follow the following instructions :
- The PreTrained models of SST-2 and IMDB can be found at:
1 - SST-2 : https://drive.google.com/file/d/1h6tWtZ-y5KkBiNok4qXZcr8YQaMHKRQ0/view
2 - IMDB  : https://drive.google.com/file/d/1v7wO1QYuWsS08kuiEhQLT5ofuzw2JhfZ/view
- For the SST-2, run /sst2/run_sst.py, this will yield the following files:
  * all_flips_pruning.p , all_flips_generate.p , conservation.p
  * These pickle files can be loaded inside the notebook "paper_plots" to generate the pertubation plots along with the conservation plot in Figure 3.
- For IMDB dataset, the starting code can be found at /imdb/run_imdb.py
  * Follow the same instructions of SST-2 dataset above.
