About
=====

Chess reinforcement learning by [AlphaGo Zero](https://deepmind.com/blog/alphago-zero-learning-scratch/) methods.

This project is based in two main resources:
1) DeepMind's Oct19th publication: [Mastering the Game of Go without Human Knowledge](https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ).
2) The <b>great</b> Reversi development of the DeepMind ideas that @mokemokechicken did in his repo: https://github.com/mokemokechicken/reversi-alpha-zero

Note: <b>This project is still under construction!!</b>

Environment
-----------

* Python 3.6.3
* tensorflow-gpu: 1.3.0
* Keras: 2.0.8

### New Supervised Learning Training Pipeline

I've done a supervised learning new pipeline step (to use those human games files "PGN" we can find in internet as play-data generator).
This SL step was also used in the first and original version of AlphaGo and maybe chess is a some complex game that we have to pre-train first the policy model before starting the self-play process (i.e., maybe chess is too much complicated for a self training alone).

To use the new SL process is so simple as running in the beginning instead of the worker "self" the new worker "sl".
Once the model converges enough with SL play-data we just stop the worker "sl" and start the worker "self" so the model will start improving now due to self-play data.

If you want to use this new SL step you will have to download from internet big PGN files (chess files) and paste them into the "data/play_data" folder.

Supervised Learning
-------------------

```bash
python src/chess_zero/run.py sl
```

### New Distributed Training Pipeline

Now it's possible to train the model in a distributed way. The only thing needed is to use the new parameter:

* `--type distributed`: use mini config for testing, (see `src/chess_zero/configs/distributed.py`)

So, in order to contribute to the distributed team you just need to run the three workers locally like this:

```bash
python src/chess_zero/run.py self --type distributed (or python src/chess_zero/run.py sl --type distributed)
python src/chess_zero/run.py opt --type distributed
python src/chess_zero/run.py eval --type distributed
```

Modules
-------

### Reinforcement Learning

This AlphaGo Zero implementation consists of three worker `self`, `opt` and `eval`.

* `self` is Self-Play to generate training data by self-play using BestModel.
* `opt` is Trainer to train model, and generate next-generation models.
* `eval` is Evaluator to evaluate whether the next-generation model is better than BestModel. If better, replace BestModel.

### Evaluation

For evaluation, you can play chess with the BestModel.

* `play_gui` is Play Game vs BestModel using ASCII character encoding.

Data
-----

* `data/model/model_best_*`: BestModel.
* `data/model/next_generation/*`: next-generation models.
* `data/play_data/play_*.json`: generated training data.
* `logs/main.log`: log file.
  
If you want to train the model from the beginning, delete the above directories.

How to use
==========

Setup
-------
### install libraries
```bash
pip install -r requirements.txt
```

If you want use GPU,

```bash
pip install tensorflow-gpu
```

### set environment variables
Create `.env` file and write this.

```text:.env
KERAS_BACKEND=tensorflow
```


Basic Usages
------------

For training model, execute `Self-Play`, `Trainer` and `Evaluator`. 


Self-Play
--------

```bash
python src/chess_zero/run.py self
```

When executed, Self-Play will start using BestModel.
If the BestModel does not exist, new random model will be created and become BestModel.

### options
* `--new`: create new BestModel
* `--type mini`: use mini config for testing, (see `src/chess_zero/configs/mini.py`)

Trainer
-------

```bash
python src/chess_zero/run.py opt
```

When executed, Training will start.
A base model will be loaded from latest saved next-generation model. If not existed, BestModel is used.
Trained model will be saved every 2000 steps(mini-batch) after epoch. 

### options
* `--type mini`: use mini config for testing, (see `src/chess_zero/configs/mini.py`)
* `--total-step`: specify total step(mini-batch) numbers. The total step affects learning rate of training. 

Evaluator
---------

```bash
python src/chess_zero/run.py eval
```

When executed, Evaluation will start.
It evaluates BestModel and the latest next-generation model by playing about 200 games.
If next-generation model wins, it becomes BestModel. 

### options
* `--type mini`: use mini config for testing, (see `src/chess_zero/configs/mini.py`)

Play Game
---------

```bash
python src/chess_zero/run.py play_gui
```


When executed, ordinary chess board will be displayed in ASCII code and you can play against BestModel.


Tips and Memo
====

GPU Memory
----------

Usually the lack of memory cause warnings, not error.
If error happens, try to change `per_process_gpu_memory_fraction` in `src/worker/{evaluate.py,optimize.py,self_play.py}`,

```python
tf_util.set_session_config(per_process_gpu_memory_fraction=0.2)
```

Less batch_size will reduce memory usage of `opt`.
Try to change `TrainerConfig#batch_size` in `NormalConfig`.


Model Performance
-------

The following table is records of the best models.

|best model generation|winning percentage to best model|Time Spent(hours)|note|
|-----|-----|-----|-----|
|1|-|-|　|

