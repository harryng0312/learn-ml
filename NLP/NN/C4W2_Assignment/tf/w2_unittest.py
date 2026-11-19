import tensorflow as tf
import numpy as np
from dlai_grader.grading import test_case, print_feedback

SEED = 10


# -

def print_feedback(test_cases):
    num_cases = len(test_cases)
    failed_cases = [t for t in test_cases if t.failed == True]
    feedback_msg = "\033[92m All tests passed!"
    if failed_cases:
        feedback_msg = ""
        for failed_case in failed_cases:
            feedback_msg += f"\033[91mFailed test case: {failed_case.msg}.\nExpected: {failed_case.want}\nGot: {failed_case.got}\n\n"
    print(feedback_msg)


def test_scaled_dot_product_attention(target):
    
    def g():
        q = np.array([[1, 1, 0, 1], [0, 1, 1, 1], [1, 0, 1, 1]]).astype(np.float32)
        k = np.array([[1, 1, 0, 1], [1, 0, 1, 1 ], [1, 1, 1, 0], [0, 0, 0, 1]]).astype(np.float32)
        v = np.array([[0, 0], [1, 0], [1, 0], [1, 1]]).astype(np.float32)
        mask = np.array([[[0, 1, 0, 1], [1, 0, 0, 1], [1, 1, 0, 1]]])

        unmasked_weights_solution = [[0.3874556,  0.23500372, 0.23500372, 0.14253697],
                                     [0.2772748,  0.2772748,  0.2772748,  0.16817565],
                                     [0.23500372, 0.3874556,  0.23500372, 0.14253697]]
        
        unmasked_output_solution = [[0.6125444,  0.14253697],
                                    [0.7227252,  0.16817565],
                                    [0.7649963,  0.14253697]]
        
        
        masked_weights_solution = [[0.0,        0.62245935, 0.0,        0.37754068],
                                   [0.62245935, 0.0,        0.0,        0.37754068],
                                   [0.30719587, 0.5064804,  0.0,        0.18632373]]
        
        
        masked_output_solution = [[1.0,        0.37754068],
                                  [0.37754068, 0.37754068],
                                  [0.6928041,  0.18632373]]
        
        cases = []
        
        # Test for unmasked
        attention, weights = target(q, k, v, None)

        t = test_case()
        if not tf.is_tensor(weights):
            t.failed = True
            t.msg = "Attention weights must be a tensor"
            t.want = "A tensor"
            t.got = type(weights)
            return [t]
        cases.append(t)

        t = test_case()
        if tuple(tf.shape(weights).numpy()) != (q.shape[0], k.shape[1]):
            t.failed = True
            t.msg = f"Wrong shape of attention weights. Expected shape: ({q.shape[0]}, {k.shape[1]})"
            t.want = (q.shape[0], k.shape[1])
            t.got = tuple(tf.shape(weights).numpy())
        cases.append(t)        

        t = test_case()
        if not np.allclose(weights, unmasked_weights_solution):
            t.failed = True
            t.msg = "Wrong unmasked attention weights"
            t.want = unmasked_weights_solution
            t.got = weights
        cases.append(t) 

        t = test_case()
        if not tf.is_tensor(attention):
            t.failed = True
            t.msg = "Output must be a tensor"
            t.want = "A tensor"
            t.got = type(attention)
            cases.append(t)
            return [t]
        cases.append(t)
        
        t = test_case()
        if tuple(tf.shape(attention).numpy()) != (q.shape[0], v.shape[1]):
            t.failed = True
            t.msg = f"Wrong shape of output. Expected shape: ({q.shape[0]}, {v.shape[1]})"
            t.want = (q.shape[0], v.shape[1])
            t.got = tuple(tf.shape(attention).numpy())
        cases.append(t)    

        t = test_case()
        if not np.allclose(attention, unmasked_output_solution):
            t.failed = True
            t.msg = "Wrong unmasked output."
            t.want = unmasked_output_solution
            t.got = attention
        cases.append(t) 
        
        # Test for masked
        attention, weights = target(q, k, v, mask)

        t = test_case()
        if not np.allclose(weights, masked_weights_solution):
            t.failed = True
            t.msg = "Wrong masked attention weights"
            t.want = masked_weights_solution
            t.got = weights
        cases.append(t)         
        
        t = test_case()
        if not np.allclose(attention, masked_output_solution):
            t.failed = True
            t.msg = "Wrong masked output"
            t.want = masked_output_solution
            t.got = attention
        cases.append(t)
        
        return cases
    
    cases = g()
    print_feedback(cases)


def test_encoderlayer(target):
    
    def g():
        tf.keras.utils.set_random_seed(SEED)
        q = np.array([[[1, 0, 1, 1], [0, 1, 0, 1], [1, 1, 0, 1]]]).astype(np.float32)
        encoder_layer1 = target(4, 2, 8)
        
        encoded_training_true = [[ 1.7252351,  -0.71911716, -0.5152278,  -0.49089006],
                                 [ 1.1150548,   0.60933375, -0.19189441, -1.5324941 ],
                                 [ 1.373918,    0.30892202, -0.29326993, -1.3895702 ]]


        encoded_training_false = [[ 1.6023562,  -0.38022822, -0.09092891, -1.1311989 ],
                                  [ 1.1299546,   0.8555256,  -0.8922295,  -1.0932508 ],
                                  [ 1.373918,    0.30892202, -0.29326993, -1.3895702 ]]
        
        encoded = encoder_layer1(q, training=True, mask=np.array([[1, 0, 1]]))

        cases = []

        t = test_case()
        if not tf.is_tensor(encoded):
            t.failed = True
            t.msg = "Wrong type. Output must be a tensor"
            t.want = "A tensor"
            t.got = type(encoded)
            return [t]
        cases.append(t)

        t = test_case()
        if tuple(tf.shape(encoded).numpy()) != (1, q.shape[1], q.shape[2]):
            t.failed = True
            t.msg = f"Wrong shape of output. Expected shape: (1, {q.shape[1]}, {q.shape[2]})"
            t.want = (1, q.shape[1], q.shape[2])
            t.got = tuple(tf.shape(encoded).numpy())
        cases.append(t)    
        
        t = test_case()
        if not np.allclose(encoded.numpy(), encoded_training_true):
            t.failed = True
            t.msg = "Wrong values when training=True"
            t.want = encoded_training_true
            t.got = encoded.numpy()
        cases.append(t)            

        encoded = encoder_layer1(q, training=True, mask=np.array([[1, 0, 1]]))

        t = test_case()
        if not np.allclose(encoded.numpy(), encoded_training_false):
            t.failed = True
            t.msg = "Wrong values when training=False"
            t.want = encoded_training_false
            t.got = encoded.numpy()
        cases.append(t)            
        return cases

    cases = g()
    print_feedback(cases)


def test_encoder(target):
    def g():
        tf.keras.utils.set_random_seed(SEED)

        embedding_dim=4

        encoderq = target(num_layers=2,
                          embedding_dim=embedding_dim,
                          num_heads=2,
                          fully_connected_dim=8,
                          input_vocab_size=32,
                          maximum_position_encoding=5)

        x = np.array([[2, 1, 1], [0, 2, 1]])

        encoderq_output = encoderq(x, training=True, mask=None)

        case_1_result = [[[ 0.59299076,  1.1811765,  -1.4698956,  -0.30427158],
                          [ 0.4527551,  -0.6035855,  -1.2321247,   1.3829551 ],
                          [ 0.6917657,  -1.2750067,  -0.6372852,   1.2205261 ]],

                         [[-0.28437185,  1.2316655,  -1.4619398,   0.5146462 ],
                          [ 0.24883842, -0.71734357, -1.0529436,   1.5214486 ],
                          [ 1.4702177,  -1.1770345,  -0.5993981,   0.30621478]]]

        case_2_result = [[[ 0.41541427,  1.156845,   -1.5776887,   0.00542951],
                          [ 0.98596674, -0.6362744,  -1.3066326,   0.95694023],
                          [ 0.6143839,  -1.1271434,  -0.79733163,  1.3100913 ]],

                         [[-0.36302108,  0.60569626, -1.4388914,   1.1962163 ],
                          [ 0.9353545,  -0.8069399,  -1.1745483,   1.0461336 ],
                          [ 0.69199055, -1.131403,   -0.81579685,  1.2552092 ]]]

        case_3_result = [[[ 0.07893151,  1.2475345,  -1.5457472,   0.21928135],
                          [ 0.93491054, -0.5919353,  -1.3369786,   0.99400324],
                          [ 0.6504845,  -1.0995319,  -0.8401536,   1.2892009 ]],

                         [[-0.00873655,  0.6826061,  -1.6228037,   0.9489342 ],
                          [ 0.695447,   -0.81193054, -1.1355624,   1.2520459 ],
                          [ 0.7177756,  -1.132439,   -0.8213581,   1.2360215 ]]]
        
        cases = []
        
        t = test_case()
        if not tf.is_tensor(encoderq_output):
            t.failed = True
            t.msg = "Wrong type. Output must be a tensor"
            t.want = "A tensor"
            t.got = type(encoderq_output)
            return [t]
        cases.append(t)
        
        t = test_case()
        if tuple(tf.shape(encoderq_output).numpy()) != (x.shape[0], x.shape[1], embedding_dim):
            t.failed = True
            t.msg = f"Wrong shape of output. Expected shape: ({x.shape[0]}, {x.shape[1]}, {embedding_dim})"
            t.want = ({x.shape[0]}, {x.shape[1]}, {embedding_dim})
            t.got = tuple(tf.shape(encoderq_output).numpy())
        cases.append(t)    
        
        t = test_case()
        if not np.allclose(encoderq_output.numpy(), case_1_result):
            t.failed = True
            t.msg = "Wrong values for test case 1"
            t.want = case_1_result
            t.got = encoderq_output.numpy()
        cases.append(t)            

        encoderq_output = encoderq(x, training=True, mask=np.array([[[[1., 1., 1.]]], [[[1., 1., 0.]]]]))

        t = test_case()
        if not np.allclose(encoderq_output.numpy(), case_2_result):
            t.failed = True
            t.msg = "Wrong values for test case 2"
            t.want = case_2_result
            t.got = encoderq_output.numpy()
        cases.append(t) 
        
        encoderq_output = encoderq(x, training=False, mask=np.array([[[[1., 1., 1.]]], [[[1., 1., 0.]]]]))

        t = test_case()
        if not np.allclose(encoderq_output.numpy(), case_3_result):
            t.failed = True
            t.msg = "Wrong values for test case 3"
            t.want = case_3_result
            t.got = encoderq_output.numpy()
        cases.append(t) 
        
        return cases

    cases = g()
    print_feedback(cases)


def test_decoderlayer(target, create_look_ahead_mask):
    def g():
        num_heads=8
        tf.keras.utils.set_random_seed(SEED)

        decoderLayerq = target(
            embedding_dim=4, 
            num_heads=num_heads,
            fully_connected_dim=32, 
            dropout_rate=0.1, 
            layernorm_eps=1e-6)

        encoderq_output = tf.constant([[[-0.4,  0.4, -1.2,   1.5],
                                       [ 0.4,   0.2, -1.6,   0.9],
                                       [ 0.4,  -1.6,  0.1,   1.2]]])

        q = np.array([[[1, 0, 1, 1], [1, 0, 1, 1], [1, 0, 1, 1]]]).astype(np.float32)

        look_ahead_mask = create_look_ahead_mask(q.shape[1])

        padding_mask = None
        out, attn_w_b1, attn_w_b2 = decoderLayerq(q, encoderq_output, training=True, look_ahead_mask=look_ahead_mask, padding_mask=padding_mask)

        shape1 = (q.shape[0], num_heads, q.shape[1], q.shape[1])
        
        cases = []
        
        t = test_case()
        if not tf.is_tensor(attn_w_b1):
            t.failed = True
            t.msg = "Wrong type for attn_w_b1. Output must be a tensor"
            t.want = "A tensor"
            t.got = type(attn_w_b1)
            return [t]
        cases.append(t)

        t = test_case()
        if not tf.is_tensor(attn_w_b2):
            t.failed = True
            t.msg = "Wrong type for attn_w_b2. Output must be a tensor"
            t.want = "A tensor"
            t.got = type(attn_w_b1)
            cases.append(t)
            return cases
        cases.append(t)
        
        t = test_case()
        if not tf.is_tensor(out):
            t.failed = True
            t.msg = "Wrong type for out. Output must be a tensor"
            t.want = "A tensor"
            t.got = type(attn_w_b1)
            cases.append(t)
            return cases
        cases.append(t)
    
        t = test_case()
        if tuple(tf.shape(attn_w_b1).numpy()) != shape1:
            t.failed = True
            t.msg = f"Wrong shape of attn_w_b1. Expected shape: {shape1}"
            t.want = shape1
            t.got = tuple(tf.shape(attn_w_b1).numpy())
        cases.append(t)  

        t = test_case()
        if tuple(tf.shape(attn_w_b2).numpy()) != shape1:
            t.failed = True
            t.msg = f"Wrong shape of attn_w_b2. Expected shape: {shape1}"
            t.want = shape1
            t.got = tuple(tf.shape(attn_w_b2).numpy())
        cases.append(t)  
        
        t = test_case()
        if tuple(tf.shape(out).numpy()) != q.shape:
            t.failed = True
            t.msg = f"Wrong shape of out. Expected shape: {q.shape}"
            t.want = q.shape
            t.got = tuple(tf.shape(out).numpy())
        cases.append(t)  

        t = test_case()
        if not np.allclose(attn_w_b1[0, 0, 1], [0.5, 0.5, 0.], atol=1e-2):
            t.failed = True
            t.msg = "Wrong values in 'attn_w_b1'. Check the call to self.mha1"
            t.want = [0.5, 0.5, 0.]
            t.got = attn_w_b1[0, 0, 1]
        cases.append(t)   
    
        t = test_case()
        if not np.allclose(attn_w_b2[0, 0, 1], [0.25345072, 0.29631895, 0.4502304]):
            t.failed = True
            t.msg = "Wrong values in 'attn_w_b2'. Check the call to self.mha2"
            t.want = [0.25345072, 0.29631895, 0.4502304]
            t.got = attn_w_b2[0, 0, 1]
        cases.append(t)      
    
        t = test_case()
        if not np.allclose(out[0, 0], [1.4810526, -1.0550661, -0.76189274, 0.33590612]):
            t.failed = True
            t.msg = "Wrong values in 'out'"
            t.want = [1.4810526, -1.0550661, -0.76189274, 0.33590612]
            t.got = out[0, 0]
        cases.append(t)  
        
        # Now let's try a example with padding mask
        padding_mask = np.array([[[1, 1, 0]]])
        out, attn_w_b1, attn_w_b2 = decoderLayerq(q, encoderq_output, training=True, look_ahead_mask=look_ahead_mask, padding_mask=padding_mask)

        t = test_case()
        if not np.allclose(out[0, 0], [1.4630506, -0.901353, -0.94766545, 0.3859678]):
            t.failed = True
            t.msg = "Wrong values in 'out' when we mask the last word. Are you passing the padding_mask to the inner functions?"
            t.want = [1.4630506, -0.901353, -0.94766545, 0.3859678]
            t.got = out[0, 0]
        cases.append(t)  
        
        return cases

    cases = g()
    print_feedback(cases)


def test_decoder(target, create_look_ahead_mask, create_padding_mask):
    def g():
        tf.keras.utils.set_random_seed(SEED)
        
        num_layers=7
        embedding_dim=4 
        num_heads=3
        fully_connected_dim=8
        target_vocab_size=33
        maximum_position_encoding=6

        x = np.array([[3, 2, 1], [2, 1, 0]])


        encoderq_output = tf.constant([[[-0.2,  0.1, -1.3,  1.0],
                                        [ 0.4,  0.6, -1.1,  0.7],
                                        [ 0.1, -1.6,  0.3,  1.1]],
                                       [[-0.7,  0.2, -1.1,  1.0],
                                        [-0.2, -0.2, -1.0,  1.2],
                                        [ 0.8, -1.1,  0.4,  1.3]]])

        look_ahead_mask = create_look_ahead_mask(x.shape[1])

        decoderk = target(num_layers,
                        embedding_dim, 
                        num_heads, 
                        fully_connected_dim,
                        target_vocab_size,
                        maximum_position_encoding)
        outd, att_weights = decoderk(x, encoderq_output, training=False, look_ahead_mask=look_ahead_mask, padding_mask=None)

        cases = []
        
        t = test_case()
        if not isinstance(att_weights, dict):
            t.failed = True
            t.msg = "Wrong type for attention_weights. Output must be a dictionary"
            t.want = type({1:2})
            t.got = type(att_weights)
            return [t]
        cases.append(t)

        t = test_case()
        if not tf.is_tensor(outd):
            t.failed = True
            t.msg = "Wrong type for x. Output must be a tensor"
            t.want = "A tensor"
            t.got = type(outd)
            cases.append(t)
            return cases
        cases.append(t)
        
        t = test_case()
        if not np.allclose(tf.shape(outd), tf.shape(encoderq_output)):
            t.failed = True
            t.msg = f"Wrong shape of x. Expected shape: {tf.shape(encoderq_output)}"
            t.want = tf.shape(encoderq_output)
            t.got = tf.shape(outd)
        cases.append(t)  

        t = test_case()
        if not np.allclose(outd[1, 1], [1.1661551, 0.59689575, -1.4855151, -0.27753586]):
            t.failed = True
            t.msg = "Wrong values in x"
            t.want = [1.1661551, 0.59689575, -1.4855151, -0.27753586]
            t.got = outd[1, 1]
        cases.append(t)   

        keys = list(att_weights.keys())

        t = test_case()
        if len(keys) != 2 * num_layers:
            t.failed = True
            t.msg = f"Wrong length for attention weights. It must be 2 x num_layers = {2*num_layers}"
            t.want = 2 * num_layers
            t.got = len(keys)
        cases.append(t)    
        
        t = test_case()
        if not tf.is_tensor(att_weights[keys[0]]):
            t.failed = True
            t.msg = f"Wrong type for att_weights[{keys[0]}]. Output must be a tensor"
            t.want = "A tensor"
            t.got = type(att_weights[keys[0]])
            cases.append(t)
            return cases
        cases.append(t)   
    
        shape1 = (x.shape[0], num_heads, x.shape[1], x.shape[1])
        
        t = test_case()
        if tuple(tf.shape(att_weights[keys[1]]).numpy()) != shape1:
            t.failed = True
            t.msg = f"Wrong shape of attention_weights[{keys[1]}]. Expected shape: {shape1}"
            t.want = shape1
            t.got = tf.shape(att_weights[keys[1]]).numpy()
        cases.append(t)  
        
        t = test_case()
        if not np.allclose(att_weights[keys[0]][0, 0, 1], [0.5049869, 0.4950131, 0.]):
            t.failed = True
            t.msg = f"Wrong values in att_weights[{keys[0]}]"
            t.want = [0.5049869, 0.4950131, 0.]
            t.got = att_weights[keys[0]][0, 0, 1]
        cases.append(t)
        
        outd, att_weights = decoderk(x, encoderq_output, training=True, look_ahead_mask=look_ahead_mask, padding_mask=None)
        
        t = test_case()
        if not np.allclose(outd[1, 1], [0.0568594, -1.0754474, -0.56806743, 1.5866553]):
            t.failed = True
            t.msg = "Wrong values in outd when training=True"
            t.want = [0.0568594, -1.0754474, -0.56806743, 1.5866553]
            t.got = outd[1, 1]
        cases.append(t)
        
        outd, att_weights = decoderk(x, encoderq_output, training=True, look_ahead_mask=look_ahead_mask, padding_mask=create_padding_mask(x))
        
        t = test_case()
        if not np.allclose(outd[1, 1], [0.18430373, -0.895833, -0.85054576, 1.562075]):
            t.failed = True
            t.msg = "Wrong values in outd when training=True and use padding mask"
            t.want = [0.18430373, -0.895833, -0.85054576, 1.562075]
            t.got = outd[1, 1]
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def test_transformer(target, create_look_ahead_mask, create_padding_mask):
    def g():
        tf.keras.utils.set_random_seed(SEED)

        num_layers = 6
        embedding_dim = 4
        num_heads = 4
        fully_connected_dim = 8
        input_vocab_size = 30
        target_vocab_size = 35
        max_positional_encoding_input = 5
        max_positional_encoding_target = 6

        transformer = target(num_layers, 
            embedding_dim, 
            num_heads, 
            fully_connected_dim, 
            input_vocab_size, 
            target_vocab_size, 
            max_positional_encoding_input,
            max_positional_encoding_target)
        
        # 0 is the padding value
        sentence_a = np.array([[2, 3, 4, 3, 0]])
        sentence__b = np.array([[1, 2, 4, 0, 0]])

        enc_padding_mask = create_padding_mask(sentence_a)
        dec_padding_mask = create_padding_mask(sentence__b)

        look_ahead_mask = create_look_ahead_mask(sentence_a.shape[1])

        summary, weights = transformer(
            sentence_a,
            sentence__b,
            training=True,
            enc_padding_mask=enc_padding_mask,
            look_ahead_mask=look_ahead_mask,
            dec_padding_mask=dec_padding_mask
        )

        cases = []        
        
        t = test_case()
        if not tf.is_tensor(summary):
            t.failed = True
            t.msg = "Wrong type for summary. Output must be a tensor"
            t.want = "Tensor"
            t.got = type(summary)
            return [t]
        cases.append(t)
        
        shape1 = (sentence_a.shape[0], max_positional_encoding_input, target_vocab_size)
        
        t = test_case()
        if tuple(tf.shape(summary).numpy()) != shape1:
            t.failed = True
            t.msg = f"Wrong shape of summary. Expected shape: {shape1}"
            t.want = shape1
            t.got = tf.shape(summary).numpy()
        cases.append(t)

        summary_example_1 = [0.02848441, 0.03627992, 0.02426082, 0.02042054, 0.03156459, 0.0317674, 0.01480251, 0.03056923]
        
        t = test_case()
        if not np.allclose(summary[0, 0, 0:8], summary_example_1):
            t.failed = True
            t.msg = "Wrong values in summary"
            t.want = summary_example_1
            t.got = summary[0, 0, 0:8]
        cases.append(t)           

        t = test_case()
        if not isinstance(weights, dict):
            t.failed = True
            t.msg = "Wrong type for attention weights. It must be a dictionary"
            t.want = type({1:2})
            t.got = type(weights)
            cases.append(t) 
            return cases
        cases.append(t)        
        
        keys = list(weights.keys())        
        t = test_case()
        if len(keys) != 2 * num_layers:
            t.failed = True
            t.msg = f"Wrong length for attention weights. It must be 2 x num_layers = {2*num_layers}"
            t.want = 2 * num_layers
            t.got = len(keys)
        cases.append(t)    
        
        t = test_case()
        if not tf.is_tensor(weights[keys[0]]):
            t.failed = True
            t.msg = f"Wrong type for attention_weights[{keys[0]}]. Output must be a tensor"
            t.want = "A tensor"
            t.got = type(weights[keys[0]])
            cases.append(t)
            return cases
        cases.append(t)   

        shape2 = (sentence_a.shape[0], num_heads, sentence_a.shape[1], sentence_a.shape[1])
        
        t = test_case()
        if tuple(tf.shape(weights[keys[0]]).numpy()) != shape2:
            t.failed = True
            t.msg = f"Wrong shape of attention_weights[{keys[0]}]. Expected shape: {shape2}"
            t.want = shape2
            t.got = tf.shape(weights[keys[0]]).numpy()
        cases.append(t)
        
        t = test_case()
        if not np.allclose(weights[keys[0]][0, 0, 1], [0.48929226, 0.5107078, 0.0, 0.0, 0.0]):
            t.failed = True
            t.msg = f"Wrong values in weights[{keys[0]}]"
            t.want = [0.48929226, 0.5107078, 0.0, 0.0, 0.0]
            t.got = weights[keys[0]][0, 0, 1]
        cases.append(t)           

        tf.keras.utils.set_random_seed(SEED)
        summary, weights = transformer(
            sentence_a,
            sentence__b,
            training=False,
            enc_padding_mask=enc_padding_mask,
            look_ahead_mask=look_ahead_mask,
            dec_padding_mask=dec_padding_mask)
        
        summary_example_2 = [0.02850271, 0.03705187, 0.02455955, 0.02084055, 0.03113986, 0.03163682, 0.01485186, 0.03140991]

        t = test_case()
        if not np.allclose(summary[0, 0, 0:8], summary_example_2):
            t.failed = True
            t.msg = "Wrong values in summary"
            t.want = summary_example_2
            t.got = summary[0, 0, 0:8]
        cases.append(t)                
        
        return cases

    cases = g()
    print_feedback(cases)


def test_next_word(target, model, encoder_input, output):
    def g():
        
        next_word = target(model, encoder_input, output)

        cases = []        
        
        t = test_case()
        if not tf.is_tensor(next_word):
            t.failed = True
            t.msg = "Wrong type for predicted_id Output must be a tensor"
            t.want = "Tensor"
            t.got = type(next_word)
            return [t]
        cases.append(t)
        
        t = test_case()
        if next_word.dtype != tf.int32:
            t.failed = True
            t.msg = f"Returned tensor should contain tf.int32 type"
            t.want = tf.int32
            t.got = next_word.dtype
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)




