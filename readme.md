[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/kmader/chess-alpha-zero/master?urlpath=lab)
[![Demo Notebook](https://img.shields.io/badge/launch-demo_notebook-red.svg)](https://mybinder.org/v2/gh/kmader/chess-alpha-zero/master?filepath=notebooks%2Fdemo.ipynb)

About
=====

Chess reinforcement learning by [AlphaGo Zero](https://deepmind.com/blog/alphago-zero-learning-scratch/) methods.

This project is based on these main resources:
1) DeepMind's Oct 19th publication: [Mastering the Game of Go without Human Knowledge](https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ).
2) The <b>great</b> Reversi development of the DeepMind ideas that @mokemokechicken did in his repo: https://github.com/mokemokechicken/reversi-alpha-zero
3) DeepMind just released a new version of AlphaGo Zero (named now AlphaZero) where they master chess from scratch:
https://arxiv.org/pdf/1712.01815.pdf. In fact, in chess AlphaZero outperformed Stockfish after just 4 hours (300k steps) Wow!

See the [wiki](https://github.com/Akababa/Chess-Zero/wiki) for more details.

Note
----

I'm the creator of this repo. I (and some others collaborators did our best: https://github.com/Zeta36/chess-alpha-zero/graphs/contributors) but we found the self-play is too much costed for an only machine. Supervised learning worked fine but we never try the self-play by itself.

Anyway I want to mention we have moved to a new repo where lot of people is working in a distributed version of AZ for chess (MCTS in C++): https://github.com/glinscott/leela-chess

Project is almost done and everybody will be able to participate just by executing a pre-compiled windows (or Linux) application. A really great job and effort has been done is this project and I'm pretty sure we'll be able to simulate the DeepMind results in not too long time of distributed cooperation.

So, I ask everybody that wish to see a UCI engine running a neural network to beat Stockfish go into that repo and help with his machine power.

Environment
-----------

* Python 3.6.3
* tensorflow-gpu: 1.3.0
* Keras: 2.0.8

### New results (after a great number of modifications due to @Akababa)

Using supervised learning on about 10k games, I trained a model (7 residual blocks of 256 filters) to a guesstimate of 1200 elo with 1200 sims/move. One of the strengths of MCTS is it scales quite well with computing power.

Here you can see an example where I (black) played against the model in the repo (white):

![img](https://user-images.githubusercontent.com/4205182/34333105-ada817c6-e8fe-11e7-8c01-5958aaf264c1.gif)

Here you can see an example of a game where I (white, ~2000 elo) played against the model in this repo (black):

![img](https://user-images.githubusercontent.com/4205182/34323276-ecd2a7b6-e806-11e7-856a-4e2394bd75df.gif)

### First "good" results

Using the new supervised learning step I created, I've been able to train a model to the point that seems to be learning the openings of chess. Also it seems the model starts to avoid losing naively pieces.

Here you can see an example of a game played for me against this model (AI plays black):

![partida1](https://user-images.githubusercontent.com/17341905/33597844-ea53c8ae-d9a0-11e7-8564-4b9b0f35a221.gif)

Here we have a game trained by @bame55 (AI plays white):

![partida3](https://user-images.githubusercontent.com/17341905/34030278-8796f7c6-e16c-11e7-9ba4-97af15f2cde5.gif)

This model plays in this way after only 5 epoch iterations of the 'opt' worker, the 'eval' worker changed 4 times the best model (4 of 5). At this moment the loss of the 'opt' worker is 5.1 (and still seems to be converging very well).

Modules
-------

### Supervised Learning

I've done a supervised learning new pipeline step (to use those human games files "PGN" we can find in internet as play-data generator).
This SL step was also used in the first and original version of AlphaGo and maybe chess is a some complex game that we have to pre-train first the policy model before starting the self-play process (i.e., maybe chess is too much complicated for a self training alone).

To use the new SL process is as simple as running in the beginning instead of the worker "self" the new worker "sl".
Once the model converges enough with SL play-data we just stop the worker "sl" and start the worker "self" so the model will start improving now due to self-play data.

```bash
python src/chess_zero/run.py sl
```
If you want to use this new SL step you will have to download big PGN files (chess files) and paste them into the `data/play_data` folder ([FICS](http://ficsgames.org/download.html) is a good source of data). You can also use the [SCID program](http://scid.sourceforge.net/) to filter by headers like player ELO, game result and more.

**To avoid overfitting, I recommend using data sets of at least 3000 games and running at most 3-4 epochs.**

### Reinforcement Learning

This AlphaGo Zero implementation consists of three workers: `self`, `opt` and `eval`.

* `self` is Self-Play to generate training data by self-play using BestModel.
* `opt` is Trainer to train model, and generate next-generation models.
* `eval` is Evaluator to evaluate whether the next-generation model is better than BestModel. If better, replace BestModel.


### Distributed Training

Now it's possible to train the model in a distributed way. The only thing needed is to use the new parameter:

* `--type distributed`: use mini config for testing, (see `src/chess_zero/configs/distributed.py`)

So, in order to contribute to the distributed team you just need to run the three workers locally like this:

```bash
python src/chess_zero/run.py self --type distributed (or python src/chess_zero/run.py sl --type distributed)
python src/chess_zero/run.py opt --type distributed
python src/chess_zero/run.py eval --type distributed
```

### GUI
* `uci` launches the Universal Chess Interface, for use in a GUI.

To set up ChessZero with a GUI, point it to `C0uci.bat` (or rename to .sh).
For example, this is screenshot of the random model using Arena's self-play feature:
![capture](https://user-images.githubusercontent.com/4205182/34057277-e9c99118-e19b-11e7-91ee-dd717f7efe9d.PNG)

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

If you want to use GPU, follow [these instructions](https://www.tensorflow.org/install/) to install with pip3.

Make sure Keras is using Tensorflow and you have Python 3.6.3+. Depending on your environment, you may have to run python3/pip3 instead of python/pip.


Basic Usage
------------

For training model, execute `Self-Play`, `Trainer` and `Evaluator`.

**Note**: Make sure you are running the scripts from the top-level directory of this repo, i.e. `python src/chess_zero/run.py opt`, not `python run.py opt`.


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
Trained model will be saved every epoch.

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


Tips and Memory
====

GPU Memory
----------

Usually the lack of memory cause warnings, not error.
If error happens, try to change `vram_frac` in `src/configs/mini.py`,

```python
self.vram_frac = 1.0
```

Smaller batch_size will reduce memory usage of `opt`.
Try to change `TrainerConfig#batch_size` in `MiniConfig`.
