import mxnet as mx

class AlexNet:

    @staticmethod
    def build_experiment_1(num_classes):
        '''Experiment 1

        Placing Batch normalization layer BEFORE activation layer.
        Using standard ReLU activation function.

        '''

        data = mx.sym.Variable("data")

        # Block 1
        conv_1_1 = mx.sym.Convolution(data=data, kernel=(11, 11), stride=(4, 4), num_filter=96)
        bn_1_1 = mx.sym.BatchNorm(data=conv_1_1)
        act_1_1 = mx.sym.Activation(data=bn_1_1, act_type="relu")
        pool_1_1 = mx.sym.Pooling(data=act_1_1, pool_type="max", kernel=(3, 3), stride=(2, 2))
        dropout_1_1 = mx.sym.Dropout(data=pool_1_1, p=0.25)

        # Block 2
        conv_2_1 = mx.sym.Convolution(data=dropout_1_1, kernel=(5, 5), pad=(2, 2), num_filter=256)
        bn_2_1 = mx.sym.BatchNorm(data=conv_2_1)
        act_2_1 = mx.sym.Activation(data=bn_2_1, act_type="relu")
        pool_2_1 = mx.sym.Pooling(data=act_2_1, pool_type="max", kernel=(3, 3), stride=(2, 2))
        dropout_2_1 = mx.sym.Dropout(data=pool_2_1, p=0.25)


        # Block 3
        conv_3_1 = mx.sym.Convolution(data=dropout_2_1, kernel=(3, 3), pad=(1, 1), num_filter=384)
        bn_3_1 = mx.sym.BatchNorm(data=conv_3_1)
        act_3_1 = mx.sym.Activation(data=bn_3_1, act_type="relu")
        conv_3_2 = mx.sym.Convolution(data=act_3_1, kernel=(3, 3), pad=(1, 1), num_filter=384)
        bn_3_2 = mx.sym.BatchNorm(data=conv_3_2)
        act_3_2 = mx.sym.Activation(data=bn_3_2, act_type="relu")
        conv_3_3 = mx.sym.Convolution(data=act_3_2, kernel=(3, 3), pad=(1, 1), num_filter=256)
        bn_3_3 = mx.sym.BatchNorm(data=conv_3_3)
        act_3_3 = mx.sym.Activation(data=bn_3_3, act_type="relu")
        pool_3_1 = mx.sym.Pooling(data=act_3_3, pool_type="max", kernel=(3, 3), stride=(2, 2))
        dropout_3_1 = mx.sym.Dropout(data=pool_3_1, p=0.25)

        # Block 4
        flatten = mx.sym.Flatten(data=dropout_3_1)
        fc_4_1 = mx.sym.FullyConnected(data=flatten, num_hidden=4096)
        bn_4_1 = mx.sym.BatchNorm(data=fc_4_1)
        act_4_1 = mx.sym.Activation(data=bn_4_1, act_type="relu")
        dropout_4_1 = mx.sym.Dropout(data=act_4_1, p=0.25)

        # Block 5
        fc_5_1 = mx.sym.FullyConnected(data=dropout_4_1, num_hidden=4096)
        bn_5_1 = mx.sym.BatchNorm(data=fc_5_1)
        act_5_1 = mx.sym.Activation(data=bn_5_1, act_type="relu")
        dropout_5_1 = mx.sym.Dropout(data=act_5_1, p=0.5)

        # Softmax
        fc_6_1 = mx.sym.FullyConnected(data=dropout_5_1, num_hidden=num_classes)
        model = mx.sym.SoftmaxOutput(data=fc_6_1, name="softmax")

        return model

    @staticmethod
    def build_experiment_2(num_classes):
        '''Experiment 2

        Placing Batch normalization layer AFTER activation layer.
        Using standard ReLU activation function.

        '''

        data = mx.sym.Variable("data")

        # Block 1
        conv_1_1 = mx.sym.Convolution(data=data, kernel=(11, 11), stride=(4, 4), num_filter=96)
        act_1_1 = mx.sym.Activation(data=conv_1_1, act_type="relu")
        bn_1_1 = mx.sym.BatchNorm(data=act_1_1)
        pool_1_1 = mx.sym.Pooling(data=bn_1_1, pool_type="max", kernel=(3, 3), stride=(2, 2))
        dropout_1_1 = mx.sym.Dropout(data=pool_1_1, p=0.25)

        # Block 2
        conv_2_1 = mx.sym.Convolution(data=dropout_1_1, kernel=(5, 5), pad=(2, 2), num_filter=256)
        act_2_1 = mx.sym.Activation(data=conv_2_1, act_type="relu")
        bn_2_1 = mx.sym.BatchNorm(data=act_2_1)
        pool_2_1 = mx.sym.Pooling(data=bn_2_1, pool_type="max", kernel=(3, 3), stride=(2, 2))
        dropout_2_1 = mx.sym.Dropout(data=pool_2_1, p=0.25)


        # Block 3
        conv_3_1 = mx.sym.Convolution(data=dropout_2_1, kernel=(3, 3), pad=(1, 1), num_filter=384)
        act_3_1 = mx.sym.Activation(data=conv_3_1, act_type="relu")
        bn_3_1 = mx.sym.BatchNorm(data=act_3_1)
        conv_3_2 = mx.sym.Convolution(data=bn_3_1, kernel=(3, 3), pad=(1, 1), num_filter=384)
        act_3_2 = mx.sym.Activation(data=conv_3_2, act_type="relu")
        bn_3_2 = mx.sym.BatchNorm(data=act_3_2)
        conv_3_3 = mx.sym.Convolution(data=bn_3_2, kernel=(3, 3), pad=(1, 1), num_filter=256)
        act_3_3 = mx.sym.Activation(data=conv_3_3, act_type="relu")
        bn_3_3 = mx.sym.BatchNorm(data=act_3_3)
        pool_3_1 = mx.sym.Pooling(data=bn_3_3, pool_type="max", kernel=(3, 3), stride=(2, 2))
        dropout_3_1 = mx.sym.Dropout(data=pool_3_1, p=0.25)

        # Block 4
        flatten = mx.sym.Flatten(data=dropout_3_1)
        fc_4_1 = mx.sym.FullyConnected(data=flatten, num_hidden=4096)
        act_4_1 = mx.sym.Activation(data=fc_4_1, act_type="relu")
        bn_4_1 = mx.sym.BatchNorm(data=act_4_1)
        dropout_4_1 = mx.sym.Dropout(data=bn_4_1, p=0.25)

        # Block 5
        fc_5_1 = mx.sym.FullyConnected(data=dropout_4_1, num_hidden=4096)
        act_5_1 = mx.sym.Activation(data=fc_5_1, act_type="relu")
        bn_5_1 = mx.sym.BatchNorm(data=act_5_1)
        dropout_5_1 = mx.sym.Dropout(data=bn_5_1, p=0.5)

        # Softmax
        fc_6_1 = mx.sym.FullyConnected(data=dropout_5_1, num_hidden=num_classes)
        model = mx.sym.SoftmaxOutput(data=fc_6_1, name="softmax")

        return model

    @staticmethod
    def build_experiment_3(num_classes):
        '''Experiment 3

        Placing Batch normalization layer AFTER activation layer.
        Using standard ELU activation function.

        '''

        data = mx.sym.Variable("data")

        # Block 1
        conv_1_1 = mx.sym.Convolution(data=data, kernel=(11, 11), stride=(4, 4), num_filter=96)
        act_1_1 = mx.sym.LeakyReLU(data=conv_1_1, act_type="elu")
        bn_1_1 = mx.sym.BatchNorm(data=act_1_1)
        pool_1_1 = mx.sym.Pooling(data=bn_1_1, pool_type="max", kernel=(3, 3), stride=(2, 2))
        dropout_1_1 = mx.sym.Dropout(data=pool_1_1, p=0.25)

        # Block 2
        conv_2_1 = mx.sym.Convolution(data=dropout_1_1, kernel=(5, 5), pad=(2, 2), num_filter=256)
        act_2_1 = mx.sym.LeakyReLU(data=conv_2_1, act_type="elu")
        bn_2_1 = mx.sym.BatchNorm(data=act_2_1)
        pool_2_1 = mx.sym.Pooling(data=bn_2_1, pool_type="max", kernel=(3, 3), stride=(2, 2))
        dropout_2_1 = mx.sym.Dropout(data=pool_2_1, p=0.25)


        # Block 3
        conv_3_1 = mx.sym.Convolution(data=dropout_2_1, kernel=(3, 3), pad=(1, 1), num_filter=384)
        act_3_1 = mx.sym.LeakyReLU(data=conv_3_1, act_type="elu")
        bn_3_1 = mx.sym.BatchNorm(data=act_3_1)
        conv_3_2 = mx.sym.Convolution(data=bn_3_1, kernel=(3, 3), pad=(1, 1), num_filter=384)
        act_3_2 = mx.sym.LeakyReLU(data=conv_3_2, act_type="elu")
        bn_3_2 = mx.sym.BatchNorm(data=act_3_2)
        conv_3_3 = mx.sym.Convolution(data=bn_3_2, kernel=(3, 3), pad=(1, 1), num_filter=256)
        act_3_3 = mx.sym.LeakyReLU(data=conv_3_3, act_type="elu")
        bn_3_3 = mx.sym.BatchNorm(data=act_3_3)
        pool_3_1 = mx.sym.Pooling(data=bn_3_3, pool_type="max", kernel=(3, 3), stride=(2, 2))
        dropout_3_1 = mx.sym.Dropout(data=pool_3_1, p=0.25)

        # Block 4
        flatten = mx.sym.Flatten(data=dropout_3_1)
        fc_4_1 = mx.sym.FullyConnected(data=flatten, num_hidden=4096)
        act_4_1 = mx.sym.LeakyReLU(data=fc_4_1, act_type="elu")
        bn_4_1 = mx.sym.BatchNorm(data=act_4_1)
        dropout_4_1 = mx.sym.Dropout(data=bn_4_1, p=0.25)

        # Block 5
        fc_5_1 = mx.sym.FullyConnected(data=dropout_4_1, num_hidden=4096)
        act_5_1 = mx.sym.LeakyReLU(data=fc_5_1, act_type="elu")
        bn_5_1 = mx.sym.BatchNorm(data=act_5_1)
        dropout_5_1 = mx.sym.Dropout(data=bn_5_1, p=0.5)

        # Softmax
        fc_6_1 = mx.sym.FullyConnected(data=dropout_5_1, num_hidden=num_classes)
        model = mx.sym.SoftmaxOutput(data=fc_6_1, name="softmax")

        return model
