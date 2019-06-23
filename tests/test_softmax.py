import os
import tempfile
from unittest import TestCase
import numpy as np
from keras_adaptive_softmax.backend import keras
from keras_adaptive_softmax.backend import backend as K
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
                        5.605619080029101e-09, 5.3742809541290626e-05, 2.656812284840271e-06,
                        0.9891002774238586, 7.272975926753134e-05, 9.171863979418049e-08,
                        4.2644995090768134e-08, 4.891299454357068e-07, 0.0001877533650258556,
                        2.692615908017615e-07, 2.0465558772404108e-13, 1.0421855435940874e-12,
                        4.763625415689907e-16, 6.248159711245028e-16, 1.2282270557124199e-17,
                        1.91092305790641e-15, 1.011179546806006e-14, 5.604534844131151e-16,
                        1.9398286416841964e-13, 2.4061240738552925e-13, 0.0024439108092337847,
                        0.00016026716912165284, 0.0027854444924741983, 2.3448970750905573e-05,
                        0.005013648886233568, 3.566274244803935e-05, 5.340438292478211e-06,
                        1.6347643395420164e-05, 4.3817875848617405e-05, 5.400039299274795e-05,
                ],
                [
                        4.812825136468746e-05, 0.9990597367286682, 5.242341103439685e-06,
                        2.8096915016817547e-08, 1.5739469745312817e-05, 0.0008400182705372572,
                        8.577504786444479e-07, 2.854927515727468e-05, 1.5727113122920855e-06,
                        2.7855088902128955e-08, 7.1702964746234166e-12, 4.627806926733868e-13,
                        2.359033146603906e-08, 1.4572094919618195e-10, 1.1793275689342408e-07,
                        2.6789748197586505e-10, 7.891776815371898e-11, 1.321358578110221e-10,
                        9.019042179758685e-10, 5.2250877236037496e-11, 1.574426100603432e-08,
                        4.194554037439957e-09, 3.735754816602821e-09, 1.7098928939418556e-09,
                        4.4564735901531094e-08, 4.4740704197021586e-12, 9.015011502057357e-13,
                        6.741370048995998e-12, 6.530154195161231e-11, 1.1295677934675119e-11,
                ],
                [
                        0.0009261462837457657, 0.0011657116701826453, 1.2146524568379391e-05,
                        3.022600182298052e-12, 3.9758965564296034e-10, 0.9977163076400757,
                        1.3056887515627835e-10, 3.617208506057068e-07, 8.730059453121441e-10,
                        2.2465255824499764e-06, 1.52069091541307e-07, 1.535757473902777e-05,
                        7.796438494800384e-11, 7.146515201839065e-08, 9.446756052966521e-07,
                        5.736067691941571e-07, 0.00015582735068164766, 9.131013278818045e-09,
                        3.47816558132763e-06, 6.498690936496132e-07, 2.6987896103005185e-14,
                        1.444208793353885e-13, 2.029298820996339e-12, 3.8475198721465986e-11,
                        3.6226284558932287e-14, 2.518819042229836e-10, 4.83924532390656e-11,
                        3.1355870677707287e-10, 2.471520987867848e-09, 5.858429852345637e-10,
                ],
                [
                        1.3934656806213752e-07, 0.9742105007171631, 1.6341533637387329e-06,
                        0.02507496066391468, 0.0002712457499001175, 2.178923023166135e-05,
                        2.0800779765295374e-08, 1.7274820720558637e-06, 0.00010041205678135157,
                        5.4775970426135245e-09, 5.6841660420409515e-14, 3.7548272596634596e-14,
                        1.1444488477556358e-13, 5.65181463506011e-15, 2.5194957746629484e-14,
                        1.0190468735233986e-14, 1.7676869489653343e-14, 5.66470054728669e-15,
                        7.503142530028428e-13, 2.5411218810590663e-13, 8.102223364403471e-05,
                        5.552934453589842e-06, 4.1136256186291575e-05, 5.870375616723322e-07,
                        0.00018922184244729578, 2.1802972760553985e-09, 4.4249195974011e-10,
                        3.3813907229784945e-09, 3.379120627755583e-08, 5.573268957448363e-09,
                ],
                [
                        0.017724955454468727, 0.9227734804153442, 0.003521926701068878,
                        7.439290357069694e-07, 0.0008589967619627714, 0.038574133068323135,
                        0.0015136339934542775, 0.011509685777127743, 0.000178004804183729,
                        0.00017167421174235642, 8.737039252082468e-07, 8.95551508506287e-08,
                        0.0005850140005350113, 1.339835853286786e-05, 0.002454278524965048,
                        3.4273638448212296e-05, 1.1094820365542546e-05, 1.0878245120693464e-05,
                        4.874425212619826e-05, 3.374404514033813e-06, 2.1057421690784395e-06,
                        8.83039831478527e-07, 6.429559107345995e-07, 7.170762614805426e-07,
                        5.868983862455934e-06, 6.818084585802353e-08, 1.142120442665373e-08,
                        4.8989768686169555e-08, 2.1333204358597868e-07, 1.252294623554917e-07,
                ],
            ],
            [
                [
                        0.019987840205430984, 0.19081535935401917, 0.004886345472186804,
                        1.745469546676759e-07, 0.009911542758345604, 0.0008648187504149973,
                        0.5589132308959961, 0.1608048975467682, 0.0006605738308280706,
                        0.000297759281238541, 3.3614483641031256e-08, 1.0848678205777063e-10,
                        0.04176267609000206, 8.236189387389459e-06, 0.011038518510758877,
                        1.4221594028640538e-05, 1.5919273721465288e-07, 1.4166244000080042e-05,
                        3.7811121273989556e-06, 1.1510935138403511e-07, 2.9287286906765075e-06,
                        6.513749326586549e-07, 7.16267294365025e-08, 7.264210211133104e-08,
                        1.1740427908080164e-05, 3.174043072817767e-08, 4.717814139354459e-09,
                        1.4121692260005148e-08, 3.665078551762235e-08, 4.744870452100258e-08,
                ],
                [
                        3.437854445564881e-08, 0.9630267024040222, 6.557394840456254e-07,
                        0.030607374384999275, 0.005104630719870329, 2.3386706970995874e-07,
                        1.8453512211635825e-06, 8.987080036604311e-06, 0.00046148369438014925,
                        1.0203468070457689e-09, 8.033208855689561e-17, 1.285992055737203e-18,
                        5.271685323514352e-13, 1.0312129824453575e-16, 4.315908151968518e-15,
                        1.0466125046475837e-16, 5.376035811752095e-18, 2.4897675863681063e-16,
                        2.5249268652174483e-15, 3.9264852838639453e-16, 0.00017593226220924407,
                        4.260230525687803e-06, 5.525957021745853e-06, 3.286311667238806e-08,
                        0.0006022685556672513, 2.0304594117170893e-10, 3.920594404682731e-11,
                        2.5789609336968056e-10, 2.0773711639776593e-09, 4.763525129902746e-10,
                ],
                [
                        0.11111080646514893, 0.4013337790966034, 0.0010328179923817515,
                        4.1730782024407276e-11, 1.2512266948760953e-05, 0.24545560777187347,
                        0.000354167161276564, 0.004766426980495453, 1.5340842764999252e-06,
                        5.2795141527894884e-05, 4.250912013503694e-07, 2.885788852324822e-08,
                        0.003187865251675248, 7.960812217788771e-05, 0.23218050599098206,
                        0.00022742505825590342, 6.903129542479292e-05, 4.830407124245539e-05,
                        8.390599396079779e-05, 2.4380628929066006e-06, 8.367688208998914e-10,
                        9.737707307166943e-10, 3.437622131396978e-10, 3.2822227158391115e-09,
                        2.3505164481463225e-09, 2.0711224402170103e-10, 3.638312567888491e-11,
                        1.80061160426348e-10, 9.632444841756183e-10, 4.1286657426198303e-10,
                ],
                [
                        1.3648341123939645e-08, 2.708638824344689e-08, 4.87171346321702e-05,
                        0.7242417335510254, 0.0021945710759609938, 1.525281151382174e-10,
                        0.00234643230214715, 6.961994949961081e-05, 0.004193916451185942,
                        9.601205965736881e-05, 1.9603370038367984e-09, 3.094915712598123e-10,
                        2.5621866850400465e-09, 5.1910288567658114e-11, 7.20926411016537e-13,
                        1.4255195635026752e-10, 1.4838219195012492e-11, 1.1114650866339559e-10,
                        1.0465009969706784e-09, 8.016137753585895e-10, 0.05781349539756775,
                        0.00217640632763505, 0.003842257894575596, 4.160598837188445e-05,
                        0.18008670210838318, 0.01127866841852665, 0.001147346687503159,
                        0.001097068190574646, 0.0005514749209396541, 0.008773954585194588,
                ],
                [
                        0.0023310172837227583, 2.793478643070557e-06, 0.04028954356908798,
                        0.00010354104597354308, 3.747308801393956e-05, 0.0016634142957627773,
                        0.0006405340973287821, 0.002110437024384737, 0.00028856395510956645,
                        0.41590359807014465, 0.0022155523765832186, 0.04603401571512222,
                        6.657961648670607e-07, 6.433246744563803e-05, 1.8276659830007702e-06,
                        0.0005191282252781093, 0.008427738212049007, 2.379375473537948e-05,
                        0.0019978247582912445, 0.001285754842683673, 5.4759566410211846e-05,
                        5.0246228056494147e-05, 0.00035087967989966273, 0.0004574395134113729,
                        9.856570977717638e-05, 0.20481349527835846, 0.023202957585453987,
                        0.030673375353217125, 0.024570459499955177, 0.1917862743139267,
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
