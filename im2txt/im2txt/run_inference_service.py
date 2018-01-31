r"""Creates a captioning RPC service."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from concurrent import futures

import math
import os
import time

import grpc

import tensorflow as tf

from im2txt import im2txt_pb2
from im2txt import im2txt_pb2_grpc

from im2txt import configuration
from im2txt import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("vocab_file", "", "Text file containing the vocabulary.")

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_PORT = 50051

session = None
vocab = None
generator = None


class Im2Txt(im2txt_pb2_grpc.Im2TxtServicer):

    def Run(self, request, context):
        captions = generator.beam_search(session, request.image)
        sentence = [vocab.id_to_word(w) for w in captions[0].sentence[1:-1]]
        sentence = " ".join(sentence)
        return im2txt_pb2.Im2TxtReply(text=sentence)


def main(_):
    print("hello")
    tf.logging.info("hello")
    # Build the inference graph.
    g = tf.Graph()
    with g.as_default():
        model = inference_wrapper.InferenceWrapper()
        restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                                   FLAGS.checkpoint_path)
    g.finalize()

    # Create the vocabulary.
    global vocab
    vocab = vocabulary.Vocabulary(FLAGS.vocab_file)

    with tf.Session(graph=g) as sess:
        global session
        global generator

        # Load the model from checkpoint.
        restore_fn(sess)

        session = sess

        # Prepare the caption generator. Here we are implicitly using the default
        # beam search parameters. See caption_generator.py for a description of the
        # available beam search parameters.
        generator = caption_generator.CaptionGenerator(model, vocab)

        tf.logging.info("server starting server on port %i" % _PORT)
        print("server starting server on port %i" % _PORT)

        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        im2txt_pb2_grpc.add_Im2TxtServicer_to_server(Im2Txt(), server)
        server.add_insecure_port('[::]:%i' % _PORT)
        server.start()
        try:
            while True:
                time.sleep(_ONE_DAY_IN_SECONDS)
        except KeyboardInterrupt:
            server.stop(0)


if __name__ == '__main__':
    tf.app.run()
