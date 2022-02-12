# Official Code Implementation of The Paper : XAI for Transformers: Better Explanations through Conservative Propagation
<p align="center">
  <img width="600" height="300" src="https://i.ibb.co/QdbXFjY/Screen-Shot-2022-02-11-at-11-26-06.png">
</p>

For the SST-2 and IMDB expermints follow the following instructions :
- The PreTrained models of SST-2 and IMDB can be found at: <br>
1 - SST-2 : https://drive.google.com/file/d/1h6tWtZ-y5KkBiNok4qXZcr8YQaMHKRQ0/view <br>
2 - IMDB  : https://drive.google.com/file/d/1v7wO1QYuWsS08kuiEhQLT5ofuzw2JhfZ/view <br>
<br>
- For reproducing the results over SST-2 dataset , please run the following : <br>

```sh
$ python /sst2/run_sst.py
```
- This will yield the following files: <br>
  * all_flips_pruning.p , all_flips_generate.p  <br>
  * These pickle files can be loaded inside the notebook "paper_plots" to generate the pertubation plots reported in the paper. <br>
- For IMDB dataset, the starting code can be found at /imdb/run_imdb.py
  * Follow the same instructions of SST-2 dataset above.
