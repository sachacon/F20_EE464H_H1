"""
TVM Timer and Tuning Interface

This timing and tuning software was make to help the progress of a
senior design project that involves testing and comparing the
inference times of image classifications on multiple models
and backends. Running this script will allow you to pick
from various platforms and backends.

Below are the supported platforms, backends, and models.
"""

supportedDevices = ['x86', 'Metal', 'Remote']
supportedPlatforms = ['MXNet', 'PyTorch', 'TensorFlow']
supportedModels = ['resnet18_v1', 'inceptionv3', 'mobilenetv2_1.0']


def get_menu(title, options=None, subtitle=None, default=-1):
    """
    A function the generates a menu that users can select from.

    :param title: The title of the menu
    :param options: The array of options for the menu. Leave blank to get a integer input
    :param subtitle: The subtitle of the menu
    :param default: The default if nothing is returned
    :return: The position in the array of the object chosen or the number inputted
    """

    from os import system

    system("clear")
    print("\n──────────────────────────── TVMUI ────────────────────────────\n")
    print(title + "\n")
    if subtitle is not None:
        print(subtitle + "\n")
    i = 1
    if options is not None:
        print("Select an option below:\n")
        for opt in options:
            print(str(i) + ") " + opt)
            i = i + 1

    inp = input("\n>> ")
    if inp == '':
        inp = default
    else:
        inp = int(inp)
    system("clear")
    if options is not None:
        if -1 < inp < i:
            return inp - 1
        else:
            return get_menu(title, options, subtitle)
    else:
        return inp


def get_pics(batch):
    """
    Get the paths of the pictures in the "pictures" folder to create a
    group divisible to batch.

    :param batch: The number the returned set should be divisible by
    :return: The array of paths to the pictures
    """
    from os import listdir
    pics = listdir("pictures/")
    nub_pics = int(int(len(pics) / batch) * batch)
    rtn = []
    i = 0
    for pic in pics:
        rtn.append("pictures/" + pic)
        i = i + 1
        if i >= nub_pics:
            break
    return rtn


def get_tunes(device, plat, mod, batch=1):
    """
    Gets the supported auto-tuned loggs for a given setup

    :param device: The supported platform
    :param plat: The supported backend
    :param mod: The supported model
    :param batch: The size of each run
    :return: The supported log paths. May return close matches that
             work if no match is found
    """
    from os import listdir
    tunes = listdir("tunings/")
    rtn = []
    for tune in tunes:
        if supportedDevices[device] in tune and supportedPlatforms[plat] in tune and supportedModels[
            mod] in tune and str(
            batch) in tune:
            rtn.append("tunings/" + tune)
        elif supportedDevices[device] in tune and supportedPlatforms[plat] in tune and supportedModels[mod] in tune:
            rtn.append("tunings/" + tune)
        elif supportedDevices[device] in tune and supportedModels[mod] in tune:
            rtn.append("tunings/" + tune)
    return rtn


def log_print(file, message):
    """
    Print to the console and a file at the same time

    :param file: The file to print to
    :param message: The message
    """
    print(message)
    file.write(message + '\n')


def get_remotes():
    """
    Gets the remote settings from remotes.json

    :return: The array of data from remotes.json
    """
    import json
    f = open("remotes.json")
    return json.load(f)


def tvm_timer():
    """
    Sets up and runes a time trail for TVM. Uses menus to get your selection
    and then runs those selections
    """
    device = get_menu("Select a supported device", supportedDevices)
    if supportedDevices[device] == "Remote":
        remotes = get_remotes()
        if len(remotes) == 0:
            get_menu("No remote devices!")
            return tvm_timer()
        remotes_names = []
        for i in remotes:
            remotes_names.append(i['name'])
        remote = remotes[get_menu("Pick a remote device", remotes_names)]
    ml_plat = get_menu("Select a machine learning platform", supportedPlatforms)
    ml_mod = get_menu("Select a pretrained model", supportedModels)
    batch = get_menu("How many pictures should be run at once? (Default 1)", default=1)
    runs = get_menu("How many times should this be run through the model (Default 3)", default=3)
    reps = get_menu("How many times should this measurement be repeated (Default 5)", default=5)
    auto = get_menu("Use AutoTVM Tunings?", ["Yes", "No"])
    tuned_log = None
    if auto == 0:
        tune_logs = get_tunes(device, ml_plat, ml_mod, batch)
        tuned_log = tune_logs[get_menu("Which file should be used for AutoTVM?", tune_logs)]
    file_name = "logs/TVMTime_" + supportedDevices[device] + "_"
    if supportedDevices[device] == "Remote":
        file_name = file_name + remote["type"] + "_"
    file_name = file_name + supportedPlatforms[ml_plat] + "_" + supportedModels[ml_mod] + "_"
    if auto == 0:
        file_name = file_name + "A"
    else:
        file_name = file_name + "N"
    file_name = file_name + str(batch) + str(runs) + str(reps)
    log = open(file_name + '.log', 'w+')
    if supportedDevices[device] == "Remote":
        run_timing(remote["hardware"], supportedPlatforms[ml_plat], supportedModels[ml_mod], remote, tuned_log, batch,
                   runs, reps, log)
    else:
        run_timing(supportedDevices[device], supportedPlatforms[ml_plat], supportedModels[ml_mod], None, tuned_log,
                   batch, runs, reps, log)


def run_timing(device, platform, model, remote=None, autotvm_log=None, batch=1, runs=3, reps=5, log=None):
    """
    Run a time trail on TVM

    :param device: The device to run this on
    :param platform: The platform get the machine learning model on
    :param model: The machine learning model to use
    :param remote: Details about the remote device
    :param autotvm_log: The path to the auto TVM file
    :param batch: The number of pictures to run in one go
    :param runs: The number of runs to run the picture through
    :param reps: The number of times the measurement should be repeated
    :param log: The output file
    """

    # Output details of run
    from cpuinfo import get_cpu_info
    from datetime import datetime

    print("\n──────────────────────────── TVMUI ────────────────────────────\n")
    log.write("TVM Time Trial\n")
    log_print(log, "Started on " + str(datetime.now().strftime("%m/%d/%Y at %H:%M:%S")))
    if remote is None:
        log_print(log, 'Hardware: ' + device)
        if device == 'x86':
            log_print(log, 'CPU Type: ' + get_cpu_info().get('brand_raw'))
    else:
        log_print(log, 'Remote Name: ' + remote["name"])
        log_print(log, 'Remote Device: ' + remote["type"])
        log_print(log, 'Remote Hardware: ' + remote["hardware"])
    log_print(log, 'Backend: ' + platform)
    log_print(log, 'Model: ' + model)
    log_print(log, str(batch) + " picture(s) per run")
    log_print(log, str(runs) + " run average, repeated " + str(reps) + " times.")
    if autotvm_log is None:
        log_print(log, 'AutoTVM: No\n')
    else:
        log_print(log, 'AutoTVM: Yes\n')

    # Get the model and image data
    import numpy as np
    from PIL import Image
    from tvm import relay
    import tvm
    from tvm.contrib.download import download_testdata

    print("Loading models and images...")

    pictures = get_pics(batch)
    dataset = []

    if platform == "MXNet":
        from mxnet.gluon.model_zoo.vision import get_model

        block = get_model(model, pretrained=True)

        synset_url = "".join(
            [
                "https://gist.githubusercontent.com/zhreshold/",
                "4d0b62f3d01426887599d4f7ede23ee5/raw/",
                "596b27d23537e5a1b5751d2b0481ef172f58b539/",
                "imagenet1000_clsid_to_human.txt",
            ]
        )
        synset_name = "imagenet1000_clsid_to_human.txt"
        synset_path = download_testdata(synset_url, synset_name, module="data")
        with open(synset_path) as f:
            synset = eval(f.read())

        def transform_image(image):
            image = np.array(image) - np.array([123.0, 117.0, 104.0])
            image /= np.array([58.395, 57.12, 57.375])
            image = image.transpose((2, 0, 1))
            image = image[np.newaxis, :]
            return image

        if model == 'resnet18_v1' or model == 'mobilenetv2_1.0':
            for img in pictures:
                dataset.append(transform_image(Image.open(img).resize((224, 224))))
            input_shape = [batch, 3, 224, 224]

        elif model == 'inceptionv3':
            for img in pictures:
                dataset.append(transform_image(Image.open(img).resize((299, 299))))
            input_shape = [batch, 3, 299, 299]
        else:
            raise Exception("Invalid Model")

        shape_dict = {"data": input_shape}

        mod, params = relay.frontend.from_mxnet(block, shape_dict)
        func = mod["main"]
        func = relay.Function(func.params, relay.nn.softmax(func.body), None, func.type_params, func.attrs)

    elif platform == "PyTorch":
        import torch
        import torchvision

        model = getattr(torchvision.models, model)(pretrained=True)
        model = model.eval()

        # We grab the TorchScripted model via tracing
        input_shape = [batch, 3, 224, 224]
        input_data = torch.randn(input_shape)
        scripted_model = torch.jit.trace(model, input_data).eval()

        synset_url = "".join(
            [
                "https://raw.githubusercontent.com/Cadene/",
                "pretrained-models.pytorch/master/data/",
                "imagenet_synsets.txt",
            ]
        )
        synset_name = "imagenet_synsets.txt"
        synset_path = download_testdata(synset_url, synset_name, module="data")
        with open(synset_path) as f:
            synsets = f.readlines()
        synsets = [x.strip() for x in synsets]
        splits = [line.split(" ") for line in synsets]
        key_to_classname = {spl[0]: " ".join(spl[1:]) for spl in splits}

        class_url = "".join(
            [
                "https://raw.githubusercontent.com/Cadene/",
                "pretrained-models.pytorch/master/data/",
                "imagenet_classes.txt",
            ]
        )
        class_name = "imagenet_classes.txt"
        class_path = download_testdata(class_url, class_name, module="data")
        with open(class_path) as f:
            class_id_to_key = f.readlines()
        class_id_to_key = [x.strip() for x in class_id_to_key]

        def transform_image(image):
            from torchvision import transforms

            my_preprocess = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            img = my_preprocess(image)
            return np.expand_dims(img, 0)

        for img in pictures:
            dataset.append(transform_image(Image.open(img).resize((224, 224))))

        input_name = "data"
        shape_list = [(input_name, input_shape)]
        func, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    elif platform == "TensorFlow":
        import tensorflow as tf
        import os

        try:
            tf_compat_v1 = tf.compat.v1
        except ImportError:
            tf_compat_v1 = tf
        import tvm.relay.testing.tf as tf_testing

        # Base location for model related files.
        repo_base = "https://github.com/dmlc/web-data/raw/main/tensorflow/models/InceptionV1/"
        model_name = "classify_image_graph_def-with_shapes.pb"
        model_url = os.path.join(repo_base, model_name)

        # Image label map
        map_proto = "imagenet_2012_challenge_label_map_proto.pbtxt"
        map_proto_url = os.path.join(repo_base, map_proto)

        # Human readable text for labels
        label_map = "imagenet_synset_to_human_label_map.txt"
        label_map_url = os.path.join(repo_base, label_map)

        model_path = download_testdata(model_url, model_name, module=["tf", "InceptionV1"])
        map_proto_path = download_testdata(map_proto_url, map_proto, module="data")
        label_path = download_testdata(label_map_url, label_map, module="data")

        with tf_compat_v1.gfile.GFile(model_path, "rb") as f:
            graph_def = tf_compat_v1.GraphDef()
            graph_def.ParseFromString(f.read())
            graph = tf.import_graph_def(graph_def, name="")
            # Call the utility to import the graph definition into default graph.
            graph_def = tf_testing.ProcessGraphDefParam(graph_def)
            # Add shapes to the graph.
            with tf_compat_v1.Session() as sess:
                graph_def = tf_testing.AddShapesToGraphDef(sess, "softmax")
        for img in pictures:
            dataset.append(np.array(Image.open(img).resize((299, 299))))
        shape_dict = {"data": [batch, 3, 299, 299]}
        dtype_dict = {"DecodeJpeg/contents": "uint8"}
        mod, params = relay.frontend.from_tensorflow(graph_def, layout=None, shape=shape_dict)
    else:
        raise Exception('Not Supported!')

    # Build the graph
    if device == 'x86':
        target = "llvm"
        ctx = tvm.cpu(0)
        log_print(log, 'Target: ' + target)
    elif device == 'Metal':
        target = "metal"
        ctx = tvm.metal(0)
        log_print(log, 'Target: ' + target)
    elif device == 'arm_cpu':
        target = tvm.target.arm_cpu(remote["type"])
        ctx = tvm.cpu(0)
        log_print(log, 'Target: ' + remote["type"])
    else:
        target = device
        ctx = tvm.cpu(0)
        log_print(log, 'Target: ' + device)
    log_print(log, 'Actual Model: ' + model + '\n')
    print('Making the graph...')
    if autotvm_log is not None:
        from tvm import autotvm
        log_print(log, 'Using AutoTVM file ' + autotvm_log)
        with autotvm.apply_graph_best(autotvm_log):
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build(func, target, params=params)
    else:
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(func, target, params=params)

    print("\nSetting up TVM...")
    from tvm.contrib import graph_runtime

    # Remote upload
    if remote is not None:
        from tvm import rpc
        from tvm.contrib import utils, graph_runtime as runtime
        print("Exporting graph...")
        tmp = utils.tempdir()
        lib_fname = tmp.relpath("net.tar")
        lib.export_library(lib_fname)
        print("Connecting to device...")
        remote = rpc.connect(str(remote["ip"]), int(remote["port"]))
        print("Uploading to device...")
        remote.upload(lib_fname)
        lib = remote.load_module("net.tar")
        if device == 'x86':
            ctx = remote.cpu(0)
        elif device == 'Metal':
            ctx = remote.metal(0)
        elif device == 'arm_cpu':
            ctx = remote.cpu(0)
        else:
            ctx = remote.cpu(0)
    dtype = "float32"
    m = graph_runtime.GraphModule(lib["default"](ctx))

    def run_tvm(pics, number, repeat):
        """
        Runs a single inference and gives back the time

        :param pics: The images(s) to run
        :param number: The number of times to run the inference
        :param repeat:  The number of times to repeat the measurement
        :return: An array with the time and the result
        """

        # combine pictures
        arr = np.ndarray(shape=input_shape, dtype=dtype)
        p = 0
        for ip in pics:
            arr[p] = ip.astype(dtype)
            p = p + 1
        m.set_input("data", tvm.nd.array(arr))

        #Actually run inference
        time = m.module.time_evaluator("run", ctx, number=number, repeat=repeat)()

        #Get output
        res = []
        if platform == 'MXNet':
            for i in range(len(pics)):
                res.append(synset[np.argmax(m.get_output(0).asnumpy()[i])])
        if platform == 'PyTorch':
            # Get top-1 result for TVM
            for i in range(len(pics)):
                top1_tvm = np.argmax(m.get_output(0).asnumpy()[i])
                tvm_class_key = class_id_to_key[top1_tvm]
                res.append(key_to_classname[tvm_class_key])
        if platform == 'TensorFlow':
            pre = np.squeeze(m.get_output(0, tvm.nd.empty(((1, 1008)), "float32")).asnumpy())
            node_lookup = tf_testing.NodeLookup(label_lookup_path=map_proto_path, uid_lookup_path=label_path)
            top_k = pre.argsort()[-5:][::-1]
            res = node_lookup.id_to_string(top_k[0])
        return [time, res]

    # Run the inferences
    output = []
    total = 0

    print("\nRunning inferences...")
    for i in range(int(len(dataset) / batch)):
        log_print(log, "\nSet " + str(i + 1) + ":")
        inp = []
        # Create the next batch
        for j in range(batch):
            inp.append(dataset[batch * i + j])
        # Run inference here
        output = run_tvm(inp, runs, reps)
        # Output results
        e = 0
        for rl in output[1]:
            log_print(log, "Image " + str(e + 1) + " Path: " + pictures[batch * i + e])
            log_print(log, "Image " + str(e + 1) + " ID: " + rl)
            e = e + 1
        log_print(log, "Time taken: " + str('%.2f' % (1000 * output[0].mean)) + " ms")
        total = total + output[0].mean
    ave = total / int(len(dataset) / batch)
    log_print(log, '\nAVERAGE TIME: ' + str(ave * 1000) + " ms")
    log_print(log, "Finished on " + str(datetime.now().strftime("%m/%d/%Y at %H:%M:%S")))
    log.close()
    return


def run_tuning():
    import os
    import numpy as np
    from tvm import autotvm
    from tvm.relay import testing
    from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
    from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner
    import tvm.contrib.graph_runtime as runtime
    from datetime import datetime

    tunemods = ["Resnet", "VGG", "MobileNet", "Squeezenet", "Inception", "MXNet"]
    tuners = ["XGBoost", "Genetic Algorithm", "Random", "Grid Search"]
    gtuners = ["DPTuner", "PBQPTuner"]

    pat = get_menu("Which platform do you want to tune?", supportedPlatforms)
    model = get_menu("Which model do you want to tune?", tunemods)
    if model == 5:
        submod = get_menu("Which submodel do you want to tune?", supportedModels)
    tunes = get_menu("Which kernel tuner do you want to use?", tuners)
    gtuner = get_menu("Which graph tuner do you want to use?", gtuners)
    batch = get_menu("How many pictures should be run at a time?")
    core = get_menu("How many cores should be used at a time?")
    print("\n──────────────────────────── TVMUI ────────────────────────────\n")
    print("Started on " + str(datetime.now().strftime("%m/%d/%Y at %H:%M:%S")))
    from tvm import relay
    import tvm
    def get_network(name, batch_size):
        """Get the symbol definition and random weight of a network"""
        input_shape = (batch_size, 3, 224, 224)
        output_shape = (batch_size, 1000)

        if "resnet" in name:
            n_layer = int(name.split("-")[1])
            mod, params = relay.testing.resnet.get_workload(
                num_layers=n_layer, batch_size=batch_size, dtype=dtype
            )
            print("Tuning ResNet")
        elif "vgg" in name:
            n_layer = int(name.split("-")[1])
            mod, params = relay.testing.vgg.get_workload(
                num_layers=n_layer, batch_size=batch_size, dtype=dtype
            )
            print("Tuning VGG")
        elif name == "mobilenet":
            mod, params = relay.testing.mobilenet.get_workload(batch_size=batch_size, dtype=dtype)
            print("Tuning MobileNet")
        elif name == "squeezenet_v1.1":
            mod, params = relay.testing.squeezenet.get_workload(
                batch_size=batch_size, version="1.1", dtype=dtype
            )
            print("Tuning SqueezeNet")
        elif name == "inception_v3":
            input_shape = (1, 3, 299, 299)
            mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
            print("Tuning Inception")
        elif name == "mxnet":
            # an example for mxnet model
            from mxnet.gluon.model_zoo.vision import get_model
            if submod == 0:
                modn = "resnet18_v1"
                print("Tuning MXNet's ResNet")
            elif submod == 1:
                modn = "inceptionv3"
                print("Tuning MXNet's Inception")
            elif submod == 2:
                modn = "mobilenetv2_1.0"
                print("Tuning MXNet's MobileNet")
            else:
                raise Exception("Not Supported!")
            block = get_model(modn, pretrained=True)
            mod, params = relay.frontend.from_mxnet(block, shape={input_name: input_shape}, dtype=dtype)
            net = mod["main"]
            net = relay.Function(
                net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs
            )
            mod = tvm.IRModule.from_expr(net)
        else:
            raise ValueError("Unsupported network: " + name)

        return mod, params, input_shape, output_shape

    if pat == 0:
        target = "llvm"
        print("Using LLVM")
    if pat == 1:
        target = "metal"
        print("Using metal")
    batch_size = batch
    if model == 0:
        dtype = "float32"
        model_name = "resnet-18"
    elif model == 1:
        dtype = "float32"
        model_name = "vgg-18"
    elif model == 2:
        dtype = "float32"
        model_name = "mobilenet"
    elif model == 3:
        dtype = "float32"
        model_name = "squeezenet_v1.1"
    elif model == 4:
        dtype = "float32"
        model_name = "inception_v3"
    elif model == 5:
        dtype = "float32"
        model_name = "mxnet"
    else:
        raise Exception('Not Supported!')
    filename = "TVMTune_" + supportedPlatforms[pat] + "_" + tunemods[model]
    if model == 5:
        filename = filename + "_" + supportedModels[submod]
    filename = filename + "_" + str(batch)
    if tunes == 0:
        filename = filename + "_XG"
    elif tunes == 1:
        filename = filename + "_GA"
    elif tunes == 2:
        filename = filename + "_RD"
    elif tunes == 3:
        filename = filename + "_GS"
    if gtuner == 0:
        filename = filename + "DP"
    elif gtuner == 1:
        filename = filename + "PB"
    log_file = "logs/" + filename + ".log"
    graph_opt_sch_file = "tunings/" + filename + "_graph_opt.log"

    # Set the input name of the graph
    # For ONNX models, it is typically "0".
    input_name = "data"

    # Set number of threads used for tuning based on the number of
    # physical CPU cores on your machine.
    num_threads = core
    os.environ["TVM_NUM_THREADS"] = str(num_threads)

    #################################################################
    # Configure tensor tuning settings and create tasks
    # -------------------------------------------------
    # To get better kernel execution performance on x86 CPU,
    # we need to change data layout of convolution kernel from
    # "NCHW" to "NCHWc". To deal with this situation, we define
    # conv2d_NCHWc operator in topi. We will tune this operator
    # instead of plain conv2d.
    #
    # We will use local mode for tuning configuration. RPC tracker
    # mode can be setup similarly to the approach in
    # :ref:`tune_relay_arm` tutorial.
    #
    # To perform a precise measurement, we should repeat the measurement several
    # times and use the average of results. In addition, we need to flush the cache
    # for the weight tensors between repeated measurements. This can make the measured
    # latency of one operator closer to its actual latency during end-to-end inference.

    tuning_option = {
        "log_filename": log_file,
        "tuner": "random",
        "early_stopping": None,
        "measure_option": autotvm.measure_option(
            builder=autotvm.LocalBuilder(),
            runner=autotvm.LocalRunner(
                number=1, repeat=10, min_repeat_ms=0, enable_cpu_cache_flush=True
            ),
        ),
    }

    # You can skip the implementation of this function for this tutorial.
    def tune_kernels(
            tasks, measure_option, tuner="gridsearch", early_stopping=None, log_filename="logs/tuning.log"
    ):

        for i, task in enumerate(tasks):
            prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

            # create tuner
            if tunes == 0:
                tuner_obj = XGBTuner(task, loss_type="rank")
                # print("Using XGBTuner")
            elif tunes == 1:
                tuner_obj = GATuner(task, pop_size=50)
                # print("Using GATuner")
            elif tunes == 2:
                tuner_obj = RandomTuner(task)
                # print("Using Random")
            elif tunes == 3:
                tuner_obj = GridSearchTuner(task)
                # print("Using GridSearch")
            else:
                raise ValueError("Invalid tuner: " + tuner)

            # do tuning
            n_trial = len(task.config_space)
            tuner_obj.tune(
                n_trial=n_trial,
                early_stopping=early_stopping,
                measure_option=measure_option,
                callbacks=[
                    autotvm.callback.progress_bar(n_trial, prefix=prefix),
                    autotvm.callback.log_to_file(log_filename),
                ],
            )

    # Use graph tuner to achieve graph level optimal schedules
    # Set use_DP=False if it takes too long to finish.
    def tune_graph(graph, dshape, records, opt_sch_file, use_DP=True):
        target_op = [
            relay.op.get("nn.conv2d"),
        ]
        if gtuner == 0:
            Tuner = DPTuner
            # print("Using DPTuner")
        else:
            Tuner = PBQPTuner
            # print("Using PBQPTuner")
        executor = Tuner(graph, {input_name: dshape}, records, target_op, target)
        executor.benchmark_layout_transform(min_exec_num=2000)
        executor.run()
        executor.write_opt_sch2record_file(opt_sch_file)

    ########################################################################
    # Finally, we launch tuning jobs and evaluate the end-to-end performance.

    def tune_and_evaluate(tuning_opt):
        # extract workloads from relay program
        print("Extract tasks...")
        mod, params, data_shape, out_shape = get_network(model_name, batch_size)
        tasks = autotvm.task.extract_from_program(
            mod["main"], target=target, params=params, ops=(relay.op.get("nn.conv2d"),)
        )

        # run tuning tasks
        tune_kernels(tasks, **tuning_opt)
        tune_graph(mod["main"], data_shape, log_file, graph_opt_sch_file)

        # compile kernels with graph-level best records
        with autotvm.apply_graph_best(graph_opt_sch_file):
            print("Compile...")
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build_module.build(mod, target=target, params=params)

            # upload parameters to device
            if pat == 0:
                ctx = tvm.cpu()
            if pat == 1:
                ctx = tvm.metal()
            data_tvm = tvm.nd.array((np.random.uniform(size=data_shape)).astype(dtype))
            module = runtime.GraphModule(lib["default"](ctx))
            module.set_input(input_name, data_tvm)

            # evaluate
            print("Evaluate inference time cost...")
            ftimer = module.module.time_evaluator("run", ctx, number=100, repeat=3)
            prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
            print(
                "Mean inference time (std dev): %.2f ms (%.2f ms)"
                % (np.mean(prof_res), np.std(prof_res))
            )

    # We do not run the tuning in our webpage server since it takes too long.
    # Uncomment the following line to run it by yourself.

    tune_and_evaluate(tuning_option)

    return


go = get_menu("Tensor Virtual Machine User Interface (TVMUI)", ["Record a time", "Create a tuning", "Exit"], "A visual "
                                                                                                             "recording "
                                                                                                             "and tuning tool "
                                                                                                             "for "
                                                                                                             "TVM")
if (go == 0):
    tvm_timer()
if (go == 1):
    run_tuning()
