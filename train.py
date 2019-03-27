import argparse
from datetime import datetime
import math
import numpy as np
import os
import subprocess
import time
import tensorflow as tf
import traceback

from datasets.datafeeder import DataFeeder
from hparams import hparams, hparams_debug_string
from models.skipNet import SkipNet 
from text import sequence_to_text
from util import audio, infolog, plot, ValueWindow
log = infolog.log


def get_git_commit():
  subprocess.check_output(['git', 'diff-index', '--quiet', 'HEAD'])   # Verify client is clean
  commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()[:10]
  log('Git commit: %s' % commit)
  return commit

def time_string():
  return datetime.now().strftime('%Y-%m-%d %H:%M')


def train(log_dir, args):
  commit = get_git_commit() if args.git else 'None'
  checkpoint_path = os.path.join(log_dir, 'model.ckpt')
  input_path = os.path.join(args.base_dir, args.input)
  log('Checkpoint path: %s' % checkpoint_path)
  log('Loading training data from: %s' % input_path)
  log('Using model: %s' % args.model)
  log(hparams_debug_string())

  # Set up DataFeeder:
  coord = tf.train.Coordinator()
  with tf.variable_scope('datafeeder') as scope:
    feeder = DataFeeder(coord, input_path, hparams)

  # Set up model:
  global_step = tf.Variable(0, name='global_step', trainable=False)
  with tf.variable_scope('model') as scope:
    model = SkipNet(hparams)
    model.initialize(feeder.txt_A, feeder.txt_A_lenth, feeder.txt_B, feeder.txt_B_lenth, \
       feeder.mel_targets, feeder.image_targets)
    model.add_loss()
    model.add_optimizer(global_step)
    stats = add_stats(model)

  # Bookkeeping:
  step = 0
  time_window = ValueWindow(100)
  loss_window = ValueWindow(100)
  saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

  # Train!
  with tf.Session() as sess:
    try:
      sess.run(tf.global_variables_initializer())

      if args.restore_step:
        # Restore from a checkpoint if the user requested it.
        restore_path = '%s-%d' % (checkpoint_path, args.restore_step)
        saver.restore(sess, restore_path)
        log('Resuming from checkpoint: %s at commit: %s' % (restore_path, commit), slack=True)
      else:
        log('Starting new training run at commit: %s' % commit, slack=True)

      feeder.start_in_session(sess)

      while not coord.should_stop():
        start_time = time.time()
        feed_dict = {self.txt_targets_A:feeder.txt_A, 
                     self.txt_lenth_A:feeder.txt_A_lenth,
                     self.txt_targets_B:feeder.txt_B, 
                     self.txt_lenth_B: feeder.txt_B_lenth, 
                     self.mel_targets: feeder.mel_targets,
                     self.image_targets: feeder.image_targets}

        # iter 1: dataset A : image - text pairs
        step, img_loss, txt_B_loss, d_loss, g_loss, opt1, opt2 =\
            sess.run(feed_dict, [global_step, model.recon_img_loss, model.recon_txt_loss_A,\
            model.domain_d_loss, model.domain_g_loss, model.optimize_recon, model.optimize_domain])

        # iter 2: dataset B: speech-text pairs
        step, speech_loss, txt_A_loss, opt1, opt2 =\
            sess.run(feed_dict, [global_step, model.recon_speech_loss, model.recon_txt_loss_B,\
            model.domain_d_loss, model.domain_g_loss, model.optimize_recon, model.optimize_domain])

 
        time_window.append(time.time() - start_time)
        loss_window.append(loss)
        message = 'Step %-7d [%.03f sec/step, loss=%.05f, avg_loss=%.05f]' % (
          step, time_window.average, loss, loss_window.average)
        log(message, slack=(step % args.checkpoint_interval == 0))

        if loss > 100 or math.isnan(loss):
          log('Loss exploded to %.05f at step %d!' % (loss, step), slack=True)
          raise Exception('Loss Exploded')

        if step % args.checkpoint_interval == 0:
          log('Saving checkpoint to: %s-%d' % (checkpoint_path, step))
          saver.save(sess, checkpoint_path, global_step=step)
          
    except Exception as e:
      log('Exiting due to exception: %s' % e, slack=True)
      traceback.print_exc()
      coord.request_stop(e)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--base_dir', default=os.getcwd())
  parser.add_argument('--input_A', default='training/train_A.txt')
  parser.add_argument('--input_B', default='training/train_B.txt')
  parser.add_argument('--model', default='m3d')
  parser.add_argument('--name', help='Name of the run. Used for logging. Defaults to model name.')
  parser.add_argument('--hparams', default='',
    help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  parser.add_argument('--restore_step', type=int, help='Global step to restore from checkpoint.')
  parser.add_argument('--checkpoint_interval', type=int, default=1000,
    help='Steps between writing checkpoints.')
  parser.add_argument('--slack_url', help='Slack webhook URL to get periodic reports.')
  parser.add_argument('--tf_log_level', type=int, default=1, help='Tensorflow C++ log level.')
  parser.add_argument('--git', action='store_true', help='If set, verify that the client is clean.')
  args = parser.parse_args()
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.tf_log_level)
  run_name = args.name or args.model
  log_dir = os.path.join(args.base_dir, 'logs-%s' % run_name)
  os.makedirs(log_dir, exist_ok=True)
  infolog.init(os.path.join(log_dir, 'train.log'), run_name, args.slack_url)
  hparams.parse(args.hparams)
  train(log_dir, args)


if __name__ == '__main__':
  main()
