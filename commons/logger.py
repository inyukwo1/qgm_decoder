import os
import datetime
import scipy.misc
import numpy as np
import logging as log
import tensorflow as tf
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x

class Logger(object):
    def __init__(self, log_dir, log_note, over_write=False):
        """Create a summary writer logging to log_dir."""
        # Check if log_dir exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Clean empty folders
        sub_dir_list = os.listdir(log_dir)
        for sub_dir in sub_dir_list:
            sub_dir_path = os.path.join(log_dir, sub_dir)
            if os.path.isdir(sub_dir_path) and not os.listdir(sub_dir_path):
                os.rmdir(sub_dir_path)

        log_note = '_' + log_note if log_note else ''
        # Make new sub directory
        self.current_time_string = datetime.datetime.now().strftime('%b_%d_%H_%M_%S')
        sub_log_dir = os.path.join(log_dir, self.current_time_string + log_note)
        if not os.path.exists(sub_log_dir):
            os.mkdir(sub_log_dir)
        log.info('tfLogging directory: ' + self.current_time_string)
        self.writer = tf.summary.FileWriter(sub_log_dir)

    def getBeginTimeString(self):
        return self.current_time_string

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()