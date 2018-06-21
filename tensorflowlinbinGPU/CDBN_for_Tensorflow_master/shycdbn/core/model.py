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

"""Abstract Class for Models"""

from abc import ABCMeta, abstractmethod, abstractproperty


class Model():
    __metaclass__ = ABCMeta

    # Build overall graphs
    @abstractmethod
    def build_graphs(self):
        pass

    # TODO: Rename
    @abstractmethod
    def build_init_ops(self):
        pass

    @abstractmethod
    def init_variables(self, sess):
        pass

    @abstractmethod
    def propagate_results(self, results):
        pass

    @abstractmethod
    def save(self, sess):
        pass

    @abstractproperty
    def input(self):
        pass

    @abstractproperty
    def output(self):
        pass

    @abstractproperty
    def ops(self):
        pass
