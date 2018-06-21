# Copyright 2016 Sanghoon Yoon
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import tensorflow as tf
from shycdbn.dataset.photos import FoursqaurePhotos
from shycdbn.core.runner import Runner
# from shycdbn.crbm import CRBM
from shycdbn.cdbn import CDBN
FLAGS = tf.app.flags.FLAGS


def main(argv=None):
    # runner = Runner(FoursqaurePhotos(),
    #                 CRBM('layer1', 300, 3, 10, 32, 2, FLAGS.batch_size, FLAGS.learning_rate, True))
   # import ipdb; ipdb.set_trace()
    runner = Runner(FoursqaurePhotos(),
                    CDBN(FLAGS.batch_size, FLAGS.learning_rate))
    runner.run()

if __name__ == '__main__':
    tf.app.run()
