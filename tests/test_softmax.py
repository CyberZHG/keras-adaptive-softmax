import os
import tempfile
from unittest import TestCase

import numpy as np
from tensorflow import keras
from tensorflow.keras import backend as K

from keras_adaptive_softmax import AdaptiveEmbedding, AdaptiveSoftmax


class AppendWeights(keras.layers.Layer):

    def __init__(self, weights, **kwargs):
        super(AppendWeights, self).__init__(**kwargs)
        self.returns = weights[:]

    def build(self, input_shape):
        for i, weight in enumerate(self.returns):
            self.returns[i] = self.add_weight(
                shape=weight.shape,
                initializer='zeros',
                name='w-{}'.format(i),
            )
        super(AppendWeights, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return [input_shape] + [K.int_shape(w) for w in self.returns]

    def compute_mask(self, inputs, mask=None):
        return [mask] + [None] * len(self.returns)

    def call(self, inputs, **kwargs):
        return [inputs] + self.returns


class TestSoftmax(TestCase):

    def test_no_projection_no_binding(self):
        input_layer = keras.layers.Input(shape=(None,))
        embed_layer = AdaptiveEmbedding(
            input_dim=3,
            output_dim=16,
            return_embeddings=True,
            return_projections=True,
        )(input_layer)
        softmax_layer = AdaptiveSoftmax(
            input_dim=16,
            output_dim=3,
        )(embed_layer)
        model = keras.models.Model(input_layer, softmax_layer)
        model_path = os.path.join(tempfile.gettempdir(), 'test_ada_softmax_%f.h5' % np.random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects={
            'AdaptiveEmbedding': AdaptiveEmbedding,
            'AdaptiveSoftmax': AdaptiveSoftmax,
        })
        model.summary()

    def test_no_projection_bind(self):
        input_layer = keras.layers.Input(shape=(None,))
        embed_layer = AdaptiveEmbedding(
            input_dim=3,
            output_dim=16,
            return_embeddings=True,
            return_projections=True,
        )(input_layer)
        softmax_layer = AdaptiveSoftmax(
            input_dim=16,
            output_dim=3,
            bind_embeddings=True,
            bind_projections=True,
        )(embed_layer)
        model = keras.models.Model(input_layer, softmax_layer)
        model_path = os.path.join(tempfile.gettempdir(), 'test_ada_softmax_%f.h5' % np.random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects={
            'AdaptiveEmbedding': AdaptiveEmbedding,
            'AdaptiveSoftmax': AdaptiveSoftmax,
        })
        model.summary()

    def test_force_projection_no_binding(self):
        input_layer = keras.layers.Input(shape=(None,))
        embed_layer = AdaptiveEmbedding(
            input_dim=3,
            output_dim=16,
            force_projection=True,
            return_embeddings=True,
            return_projections=True,
        )(input_layer)
        softmax_layer = AdaptiveSoftmax(
            input_dim=16,
            output_dim=3,
            force_projection=True,
        )(embed_layer)
        model = keras.models.Model(input_layer, softmax_layer)
        model_path = os.path.join(tempfile.gettempdir(), 'test_ada_softmax_%f.h5' % np.random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects={
            'AdaptiveEmbedding': AdaptiveEmbedding,
            'AdaptiveSoftmax': AdaptiveSoftmax,
        })
        model.summary()

    def test_force_projection_bind(self):
        input_layer = keras.layers.Input(shape=(None,))
        embed_layer = AdaptiveEmbedding(
            input_dim=3,
            output_dim=16,
            force_projection=True,
            return_embeddings=True,
            return_projections=True,
        )(input_layer)
        softmax_layer = AdaptiveSoftmax(
            input_dim=16,
            output_dim=3,
            force_projection=True,
            bind_embeddings=True,
            bind_projections=True,
        )(embed_layer)
        model = keras.models.Model(input_layer, softmax_layer)
        model_path = os.path.join(tempfile.gettempdir(), 'test_ada_softmax_%f.h5' % np.random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects={
            'AdaptiveEmbedding': AdaptiveEmbedding,
            'AdaptiveSoftmax': AdaptiveSoftmax,
        })
        model.summary()

    def test_single_projection_no_binding(self):
        input_layer = keras.layers.Input(shape=(None,))
        embed_layer = AdaptiveEmbedding(
            input_dim=3,
            output_dim=16,
            embed_dim=5,
            return_embeddings=True,
            return_projections=True,
        )(input_layer)
        softmax_layer = AdaptiveSoftmax(
            input_dim=16,
            output_dim=3,
            embed_dim=5,
        )(embed_layer)
        model = keras.models.Model(input_layer, softmax_layer)
        model_path = os.path.join(tempfile.gettempdir(), 'test_ada_softmax_%f.h5' % np.random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects={
            'AdaptiveEmbedding': AdaptiveEmbedding,
            'AdaptiveSoftmax': AdaptiveSoftmax,
        })
        model.summary()

    def test_single_projection_bind(self):
        input_layer = keras.layers.Input(shape=(None,))
        embed_layer = AdaptiveEmbedding(
            input_dim=3,
            output_dim=16,
            embed_dim=5,
            return_embeddings=True,
            return_projections=True,
        )(input_layer)
        softmax_layer = AdaptiveSoftmax(
            input_dim=16,
            output_dim=3,
            embed_dim=5,
            bind_embeddings=True,
            bind_projections=True,
        )(embed_layer)
        model = keras.models.Model(input_layer, softmax_layer)
        model_path = os.path.join(tempfile.gettempdir(), 'test_ada_softmax_%f.h5' % np.random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects={
            'AdaptiveEmbedding': AdaptiveEmbedding,
            'AdaptiveSoftmax': AdaptiveSoftmax,
        })
        model.summary()

    def test_cutoffs_no_binding(self):
        input_layer = keras.layers.Input(shape=(None,))
        embed_layer = AdaptiveEmbedding(
            input_dim=30,
            output_dim=3,
            embed_dim=8,
            cutoffs=[10, 20, 25],
            div_val=2,
            mask_zero=True,
            return_embeddings=True,
            return_projections=True,
        )(input_layer)
        softmax_layer = AdaptiveSoftmax(
            input_dim=3,
            output_dim=30,
            embed_dim=8,
            cutoffs=[10, 20, 25],
            div_val=2,
        )(embed_layer)
        model = keras.models.Model(input_layer, softmax_layer)
        model_path = os.path.join(tempfile.gettempdir(), 'test_ada_softmax_%f.h5' % np.random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects={
            'AdaptiveEmbedding': AdaptiveEmbedding,
            'AdaptiveSoftmax': AdaptiveSoftmax,
        })
        model.summary()

    def test_cutoffs_bind(self):
        input_layer = keras.layers.Input(shape=(None,))
        embed_layer = AdaptiveEmbedding(
            input_dim=30,
            output_dim=3,
            embed_dim=8,
            cutoffs=[10, 20, 25],
            div_val=2,
            mask_zero=True,
            return_embeddings=True,
            return_projections=True,
        )(input_layer)
        softmax_layer = AdaptiveSoftmax(
            input_dim=3,
            output_dim=30,
            embed_dim=8,
            cutoffs=[10, 20, 25],
            div_val=2,
            bind_embeddings=True,
            bind_projections=True,
        )(embed_layer)
        model = keras.models.Model(input_layer, softmax_layer)
        model_path = os.path.join(tempfile.gettempdir(), 'test_ada_softmax_%f.h5' % np.random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects={
            'AdaptiveEmbedding': AdaptiveEmbedding,
            'AdaptiveSoftmax': AdaptiveSoftmax,
        })
        model.summary()

    def test_cutoffs_no_projection_no_binding(self):
        input_layer = keras.layers.Input(shape=(None,))
        embed_layer = AdaptiveEmbedding(
            input_dim=30,
            output_dim=8,
            cutoffs=[10, 20, 25],
            div_val=2,
            mask_zero=True,
            force_projection=False,
            return_embeddings=True,
            return_projections=True,
        )(input_layer)
        softmax_layer = AdaptiveSoftmax(
            input_dim=8,
            output_dim=30,
            cutoffs=[10, 20, 25],
            div_val=2,
            force_projection=False,
        )(embed_layer)
        model = keras.models.Model(input_layer, softmax_layer)
        model_path = os.path.join(tempfile.gettempdir(), 'test_ada_softmax_%f.h5' % np.random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects={
            'AdaptiveEmbedding': AdaptiveEmbedding,
            'AdaptiveSoftmax': AdaptiveSoftmax,
        })
        model.summary()

    def test_cutoffs_no_projection_bind(self):
        input_layer = keras.layers.Input(shape=(None,))
        embed_layer = AdaptiveEmbedding(
            input_dim=30,
            output_dim=8,
            cutoffs=[10, 20, 25],
            div_val=2,
            mask_zero=True,
            force_projection=False,
            return_embeddings=True,
            return_projections=True,
        )(input_layer)
        softmax_layer = AdaptiveSoftmax(
            input_dim=8,
            output_dim=30,
            cutoffs=[10, 20, 25],
            div_val=2,
            force_projection=False,
            bind_embeddings=True,
            bind_projections=True,
        )(embed_layer)
        model = keras.models.Model(input_layer, softmax_layer)
        model_path = os.path.join(tempfile.gettempdir(), 'test_ada_softmax_%f.h5' % np.random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects={
            'AdaptiveEmbedding': AdaptiveEmbedding,
            'AdaptiveSoftmax': AdaptiveSoftmax,
        })
        model.summary()

    def test_sample(self):
        embed_0 = np.array([
            [
                0.7562694862279867, -0.7532437781410828, -0.2882295795429552, -1.6990371818805843,
                -0.09864164817566004, -0.5235034477186453, -1.600153091413999, 0.03441732751250957,
            ],
            [
                -0.3680529905261407, 1.1673600332887637, -0.6914459306809843, -0.7645030146906124,
                2.0434827620248606, -0.2743642839675437, 0.04834288951969495, -1.0368596183756285,
            ],
            [
                -0.8440324158987662, 0.05585795322288273, -0.5827731797867599, 1.502853709909658,
                -0.09311037618863122, 1.366316512453695, -0.3834091917878978, -1.2647642860801802,
            ],
            [
                1.5212768184170435, -0.7854311748221854, -0.4674213048014483, -1.0460200278367862,
                0.3705555995848165, -0.12273261562651422, 1.8138708310050653, -0.26957084415202764,
            ],
            [
                -0.15162771245260723, -0.19654664890167275, -1.77930041719533, -0.6987101769248606,
                0.32681036318004547, 0.19156716698736181, 0.8386004334587568, -1.8390076172747616,
            ],
            [
                -1.1363779747587972, -0.15233131547247872, 0.158423477487577, -0.6984487776859649,
                1.2424950830966563, -0.16130616338419873, -1.6298737099566283, 1.7229575808498785,
            ],
            [
                0.613169803410901, -1.5391239758406403, -1.2476893436624792, -0.05514513857644962,
                -0.5537408608863357, -0.9965187549427492, -0.6842234254089083, -1.2420165307152238,
            ],
            [
                -0.4086071455923046, -0.7286151488450243, 1.2938629380821804, 0.7450912596769113,
                -0.13042129128885002, -1.4269400640112133, -0.713571658756806, -0.5036154349645774,
            ],
            [
                0.7326026846217363, 0.12752591749386982, 0.7387086112901271, -1.4161019970745967,
                -0.6396944907214142, -2.0010110577965783, 0.5843029435066284, -0.4033331631189724,
            ],
            [
                1.22301664512685, -0.024541032664251092, -0.27128167541306714, 1.910258142043872,
                -0.9673069099782774, 0.6614265651081772, -1.165650716838653, -0.5085143504562967,
            ],
        ])
        embed_1 = np.array([
            [0.6593494357199338, -0.06233478795012013, 0.3394579881849406, 0.05894554241531747],
            [1.0015451559801243, 0.7487130375684998, -0.4244371286817957, -0.45182923128222996],
            [-0.41965070720383035, -0.2875756074838825, 1.8712603426351773, 2.531083895835167],
            [-0.6800689195006436, -0.39454047242128376, 0.5442439581019961, -0.21672610899025968],
            [-1.3119449289237803, 1.5645034642903253, 1.3203132828621442, 1.7673879116655695],
            [-0.8817194029613362, -0.6655645822150862, 0.2341787847442309, -0.7641095447924122],
            [-0.47497798682688624, 1.0109350638555383, -0.5514102704837403, -0.1450007600387442],
            [-0.531267085230172, 0.12862169808408846, 0.18339345878624577, 1.5279135983387981],
            [0.43338928943049837, 0.2660771849859784, 1.4227633495535283, -0.5072818940455809],
            [0.8704222505796531, 0.9361117741463981, 0.7442665348863866, 0.91392694614948],
        ])
        embed_2 = np.array([
            [1.2712292341556446, 1.009655780936284],
            [0.4420362222435132, 1.5186087787070979],
            [-0.10018465175352317, -0.09182475290216006],
            [-1.246047485363712, 1.6404603895987184],
            [1.4427767754835976, 1.2102150762070925],
        ])
        embed_3 = np.array([
            [0.8285545743394414],
            [0.7111875779008273],
            [0.35799413043562894],
            [-0.15005629449852656],
            [0.6263946579941496],
        ])
        proj_0 = np.array([
            [0.3409731658714878, 0.032745006392315756, 0.668797744010083],
            [-0.3082491589087075, -1.0028023345331745, 0.2122102239605163],
            [-0.3751562822576601, -0.5825445529201775, 0.43389258576225614],
            [0.26067868083146517, 0.8192897299406429, 0.073726048897453],
            [1.1346146882950412, -2.456072992985481, -0.054474463562940736],
            [-1.0283521269636255, -0.1983876737118115, 1.0132159972212373],
            [2.72334361610427, 0.5683724225575054, 2.403638230905517],
            [-0.2137114185905606, 0.3048293347650425, 1.510425235737199],
        ])
        proj_1 = np.array([
            [0.42186259731067743, 0.6034344571434473, 2.362015513199549],
            [-0.9313583984951119, -0.8242699945665621, 0.2596454482698166],
            [0.8871149648450185, -0.663397984939589, -1.195129355668761],
            [0.8016784490871957, 0.13830808473255815, -0.6580242457235711],
        ])
        proj_2 = np.array([
            [1.4802477891158519, 0.12638370704617574, -0.18503256737397666],
            [-0.3900434531439191, 0.14771223879593204, -0.8863321455068343],
        ])
        proj_3 = np.array([[-0.589729339138385, 2.018799784975004, -0.08431336326635828]])

        cluster_kernel = np.array([
            [0.23014518853189528, -1.907450615160342, -0.5760690735239482],
            [0.15888698361555206, 0.16596164514332107, -1.3874452811285074],
            [-0.43498605862409073, -0.9533547594248867, 1.376861108688103],
            [2.0713086892043306, 0.3189268504371047, 0.17466615249716405],
            [-0.995949973463762, 0.043604908747614204, -1.6117698906413622],
            [0.6066490394919954, -0.5688549027107426, 0.4277926952413127],
            [-0.045942286252255375, 1.269447988095889, -2.0882415532533565],
            [0.8578254069980026, 0.6013204537529426, -1.5562555397638154],
        ])
        cluster_bias = np.array([-1.2832837684769247, -0.39755882729529285, -1.6266054548863331])

        bias_0 = np.array([
            -0.44961683466248237, 1.1354573774120789, 1.2744817355039493, -1.5496828034299275, -0.21836162127739225,
            -0.37565781060494785, -0.17156518058295334, 0.983434075647771, -0.3062002489865936, 0.12998179587118727,
        ])
        bias_1 = np.array([
            -0.2091536758128447, -0.6470589074952924, 0.3477127052723791, -0.9045321990801439, -0.21297856640874574,
            0.3459416954179376, 0.37443354120881234, -1.1654497053299575, 1.6909447574735716, 0.23900953544990225,
        ])
        bias_2 = np.array([
            0.3099136565556444, -0.9158122257114607, -0.16860676319583162, -1.2395468248816244, 1.204462915844038,
        ])
        bias_3 = np.array([
            1.291426908829937, -0.6694533566338665, 0.2625003902625795, 0.9797242029047042, 1.599378867487272
        ])

        inputs = np.array([
            [
                [0.744236572859694, 0.016713611741487267, 1.4682734369173418],
                [0.27153908377796215, -1.469963926716969, -0.8287408146483969],
                [-2.12471847178894, -1.908653889589087, 0.6152713069444428],
                [0.9054803804104959, -1.2524010188123476, 0.673952005987055],
                [-0.05409017774217415, -0.7869076720861053, -0.8608013367536177],
            ],
            [
                [0.5928070143642264, -0.1365080521672495, -1.8938283201202142],
                [1.8238080368340632, -0.8134981522315549, -0.2736867043672396],
                [-0.6324104033897957, -1.1823330727729813, -1.4800297849679227],
                [1.3222282804156642, 1.7723967951065012, 0.38944790892928965],
                [-0.9808710814446125, 0.6626326119592982, 0.8039459587763045],
            ],
        ])

        weights = [
            embed_0, embed_1, embed_2, embed_3,
            proj_0, proj_1, proj_2, proj_3,
        ]

        input_layer = keras.layers.Input(shape=(None, 3))
        append_layer = AppendWeights(weights)
        softmax_layer = AdaptiveSoftmax(
            input_dim=3,
            output_dim=30,
            embed_dim=8,
            cutoffs=[10, 20, 25],
            div_val=2,
            bind_embeddings=True,
            bind_projections=True,
        )
        func = K.function([input_layer], [softmax_layer(append_layer(input_layer))])
        append_layer.set_weights(weights)
        softmax_layer.set_weights([cluster_kernel, cluster_bias, bias_0, bias_1, bias_2, bias_3])
        predicted = func([inputs])[0]
        expected = np.array([
            [
                [
                    5.605619080029101e-09, 5.3742809541290626e-05, 2.6568095563561656e-06,
                    0.9891002774238586, 7.272975926753134e-05, 9.171863979418049e-08,
                    4.264499864348181e-08, 4.891299454357068e-07, 0.0001877533650258556,
                    2.692615908017615e-07, 1.873376459116116e-05, 9.539959137327969e-05,
                    4.360527228186584e-08, 5.719440565599143e-08, 1.124294546350768e-09,
                    1.749220928104478e-07, 9.25613619529031e-07, 5.130279845388941e-08,
                    1.775680539140012e-05, 2.2025182261131704e-05, 0.0024439117405563593,
                    0.0001602671982254833, 0.002785446122288704, 2.3448987121810205e-05,
                    0.005013651214540005, 3.8959341395333746e-13, 5.834099344034435e-14,
                    1.785878980505376e-13, 4.786831738282094e-13, 5.899208530522893e-13,
                ],
                [
                    4.812825136468746e-05, 0.9990597367286682, 5.242341103439685e-06,
                    2.8096915016817547e-08, 1.5739469745312817e-05, 0.0008400182705372572,
                    8.577513312957308e-07, 2.8549273338285275e-05, 1.5727113122920855e-06,
                    2.7855088902128955e-08, 4.444893905659929e-15, 2.8687949779297045e-16,
                    1.4623736249719244e-11, 9.033296029595239e-14, 7.310696492623947e-11,
                    1.6607075686413814e-13, 4.8921424636999555e-14, 8.19115215651249e-14,
                    5.590938953123348e-13, 3.239051618608192e-14, 1.574426100603432e-08,
                    4.194554925618377e-09, 3.735754816602821e-09, 1.7098933380310655e-09,
                    4.4564735901531094e-08, 7.2173618193005495e-09, 1.4542597126521173e-09,
                    1.0874863676235691e-08, 1.0534140670870329e-07, 1.822166062481756e-08,
                ],
                [
                    0.000926146749407053, 0.001165713299997151, 1.2146524568379391e-05,
                    3.022600182298052e-12, 3.9759040504350196e-10, 0.9977163076400757,
                    1.305691249564589e-10, 3.6172119166621997e-07, 8.730076106466811e-10,
                    2.2465255824499764e-06, 3.152966386241185e-12, 3.184204844242089e-10,
                    1.6164958877484727e-15, 1.4817423546822917e-12, 1.9586689908868138e-11,
                    1.1893032565712947e-11, 3.2308891118049132e-09, 1.8932036114586298e-13,
                    7.211550107077969e-11, 1.3474238218236234e-11, 2.6987896103005185e-14,
                    1.444208793353885e-13, 2.029298820996339e-12, 3.8475198721465986e-11,
                    3.6226284558932287e-14, 1.2148435416747816e-05, 2.334001010240172e-06,
                    1.5123150660656393e-05, 0.00011920313409063965, 2.8255606594029814e-05,
                ],
                [
                    1.3934656806213752e-07, 0.9742105007171631, 1.6341533637387329e-06,
                    0.02507493644952774, 0.0002712457499001175, 2.178921022277791e-05,
                    2.0800779765295374e-08, 1.7274820720558637e-06, 0.00010041205678135157,
                    5.4775970426135245e-09, 2.01842809133268e-09, 1.3333264492487729e-09,
                    4.06389899509918e-09, 2.0069401696076739e-10, 8.946644536322879e-10,
                    3.6186006968641493e-10, 6.276996145082592e-10, 2.0115159538036664e-10,
                    2.6643403927550935e-08, 9.023438884980806e-09, 8.10222263680771e-05,
                    5.552933998842491e-06, 4.113625254831277e-05, 5.870374479854945e-07,
                    0.00018922182789538056, 6.140000210737642e-14, 1.2461146379355321e-14,
                    9.52243519496479e-14, 9.516042467905272e-13, 1.5695048884677154e-13,
                ],
                [
                    0.017724955454468727, 0.9227734804153442, 0.003521926701068878,
                    7.439290357069694e-07, 0.0008589967619627714, 0.03857409581542015,
                    0.0015136339934542775, 0.011509685777127743, 0.000178004804183729,
                    0.00017167421174235642, 1.2908007995804383e-10, 1.3230781054085483e-11,
                    8.642934545832759e-08, 1.9794590411237323e-09, 3.62592459168809e-07,
                    5.063550023720609e-09, 1.6391373813817722e-09, 1.607139976655958e-09,
                    7.201423013469821e-09, 4.98530883241699e-10, 2.105740350089036e-06,
                    8.830390925140819e-07, 6.429553423004108e-07, 7.17075693046354e-07,
                    5.8689788602350745e-06, 0.00046149574336595833, 7.730671495664865e-05,
                    0.0003315970825497061, 0.0014439808437600732, 0.0008476407965645194,
                ],
            ],
            [
                [
                    0.019987842068076134, 0.19081537425518036, 0.00488634780049324,
                    1.745469688785306e-07, 0.009911534376442432, 0.000864819681737572,
                    0.5589132905006409, 0.1608048379421234, 0.0006605738890357316,
                    0.00029775931034237146, 8.567404191682851e-14, 2.7650286773307244e-16,
                    1.0644154002648065e-07, 2.0991774291045928e-11, 2.813413679803034e-08,
                    3.624691866099816e-11, 4.0573834981898205e-13, 3.6105844009037824e-11,
                    9.637011674779039e-12, 2.933819907933316e-13, 2.9287286906765075e-06,
                    6.513750463454926e-07, 7.162674364735722e-08, 7.26421092167584e-08,
                    1.1740429727069568e-05, 0.01245346013456583, 0.0018510496011003852,
                    0.005540691316127777, 0.014380054548382759, 0.018616652116179466,
                ],
                [
                    3.4378548008362486e-08, 0.9630267024040222, 6.557401661666518e-07,
                    0.030607404187321663, 0.005104635842144489, 2.3386729708363418e-07,
                    1.8453529264661483e-06, 8.987089131551329e-06, 0.00046148416004143655,
                    1.0203488054472132e-09, 4.58596648069548e-13, 7.34142005406847e-15,
                    3.0094788883161527e-09, 5.886948372356426e-13, 2.4638501308626992e-11,
                    5.974854942400465e-13, 3.0690504120283596e-14, 1.4213487158076799e-12,
                    1.4414204205226433e-11, 2.241537684632977e-12, 0.00017593242228031158,
                    4.26023416366661e-06, 5.525962023966713e-06, 3.286314864681117e-08,
                    0.0006022691377438605, 3.5567560588379774e-14, 6.867705443371714e-15,
                    4.5175658596992643e-14, 3.63893133291035e-13, 8.34426725306904e-14,
                ],
                [
                    0.11111080646514893, 0.4013337790966034, 0.0010328179923817515,
                    4.1730782024407276e-11, 1.2512266948760953e-05, 0.24545560777187347,
                    0.000354167161276564, 0.004766426980495453, 1.5340842764999252e-06,
                    5.2795141527894884e-05, 3.2432775083336913e-15, 2.201742370634856e-16,
                    2.4322148461930482e-11, 6.073784595932163e-13, 1.7714454347839137e-09,
                    1.73516311995775e-12, 5.266814545254461e-13, 3.685409453741545e-13,
                    6.401695037787369e-13, 1.8601455079492353e-14, 8.367688208998914e-10,
                    9.737708417389968e-10, 3.437621853841222e-10, 3.2822229378837164e-09,
                    2.3505164481463225e-09, 0.0271458700299263, 0.0047686779871582985,
                    0.023600326851010323, 0.12625090777873993, 0.054113760590553284,
                ],
                [
                    1.3648338459404386e-08, 2.7086382914376372e-08, 4.8717127356212586e-05,
                    0.7242416143417358, 0.002194569678977132, 1.5252779594909782e-10,
                    0.0023464339319616556, 6.96199422236532e-05, 0.004193915985524654,
                    9.601204510545358e-05, 0.006397495046257973, 0.0010100157232955098,
                    0.008361614309251308, 0.00016940751811489463, 2.352720002818387e-06,
                    0.0004652136121876538, 4.8424004489788786e-05, 0.0003627230762504041,
                    0.003415221581235528, 0.002616040175780654, 0.05781353637576103,
                    0.0021764079574495554, 0.0038422606885433197, 4.1606021113693714e-05,
                    0.18008683621883392, 3.456036790083772e-09, 3.515727153846626e-10,
                    3.361662892498174e-10, 1.689842571428457e-10, 2.6885360604467223e-09,
                ],
                [
                    0.0023310158867388964, 2.793478643070557e-06, 0.04028953239321709,
                    0.00010354104597354308, 3.747312803170644e-05, 0.0016634142957627773,
                    0.0006405340973287821, 0.002110437024384737, 0.0002885640715248883,
                    0.41590359807014465, 0.01737625151872635, 0.3610380291938782,
                    5.221741503191879e-06, 0.0005045500583946705, 1.4334115803649183e-05,
                    0.004071446601301432, 0.06609751284122467, 0.00018661090871319175,
                    0.01566864363849163, 0.010083985514938831, 5.475956277223304e-05,
                    5.0246228056494147e-05, 0.00035087967989966273, 0.0004574395134113729,
                    9.8565717053134e-05, 0.026114653795957565, 0.0029584828298538923,
                    0.003910995088517666, 0.003132845275104046, 0.024453623220324516,
                ],
            ],
        ])
        self.assertTrue(np.allclose(expected, predicted))
        sums = np.sum(predicted, axis=-1)
        self.assertTrue(np.allclose(np.ones_like(sums), sums))

    def test_fit(self):
        input_layer = keras.layers.Input(shape=(None,))
        embed_layer = AdaptiveEmbedding(
            input_dim=30,
            output_dim=32,
            cutoffs=[5, 15, 25],
            div_val=2,
            return_embeddings=True,
            return_projections=True,
            mask_zero=True,
        )(input_layer)
        dense_layer = keras.layers.Dense(
            units=32,
            activation='tanh',
        )(embed_layer[0])
        softmax_layer = AdaptiveSoftmax(
            input_dim=32,
            output_dim=30,
            cutoffs=[5, 15, 25],
            div_val=2,
            bind_embeddings=True,
            bind_projections=True,
        )([dense_layer] + embed_layer[1:])
        model = keras.models.Model(inputs=input_layer, outputs=softmax_layer)
        model.compile('adam', 'sparse_categorical_crossentropy')
        model.summary()

        inputs = np.random.randint(0, 30, (4096, 10))
        outputs = np.expand_dims(inputs, axis=-1)
        model.fit(
            inputs,
            outputs,
            epochs=100,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-4, patience=2),
            ],
        )

        model = keras.models.Model(input_layer, softmax_layer)
        model_path = os.path.join(tempfile.gettempdir(), 'test_ada_softmax_%f.h5' % np.random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects={
            'AdaptiveEmbedding': AdaptiveEmbedding,
            'AdaptiveSoftmax': AdaptiveSoftmax,
        })

        inputs = np.random.randint(0, 30, (128, 10))
        outputs = model.predict(inputs).argmax(axis=-1)
        outputs *= np.not_equal(inputs, 0).astype('int32')
        diff = np.sum(np.not_equal(inputs, outputs))
        self.assertLess(diff, 5)
