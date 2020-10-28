# TVMTimer Code
# Copied most of this from tvm.apache.org/docs

supportedPlatforms = ['x86']
supportedBackend = ['MXNet', 'PyTorch', 'TensorFlow']
supportedModels = ['Resnet', 'InceptionV1', 'MobileNet']

# pictures = ["pictures/roland.jpg", "pictures/whiteclaw.jpg", "pictures/riley.jpg", "pictures/kanye.jpg"]

# Create the menu interface
def get_menu(title, options=None, subtitle=None, default=-1):
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
    from os import listdir
    pics = listdir("pictures/")
    nubpics = int(int(len(pics)/batch)*batch)
    rtn = []
    i = 0
    for pic in pics:
        rtn.append("pictures/" + pic)
        i = i+1
        if i >= nubpics:
            break
    return rtn
# run timings
def log_print(file, message):
    print(message)
    file.write(message + '\n')

def run_timing():

    plat = get_menu("Pick a platform from below", supportedPlatforms)
    back = get_menu("Pick a backend from below", supportedBackend)
    md = get_menu("Pick a model from below", supportedModels)
    auto = get_menu("Use AutoTVM Tunings?", ["Yes", "No"])
    batch = get_menu("Number of pictures to run at once? (Default 1)", default=1)
    run = get_menu("Number of times to run this inference for taking average (Default 3)", default=3)
    reps = get_menu("Number of times to repeat this measurement (Default 5)", default=5)
    filenm = "logs/TVMTime_" + supportedPlatforms[plat] + "_" + supportedModels[back] + "_" + supportedModels[md] + "_"
    if auto == 0:
        filenm = filenm + "A"
    else:
        filenm = filenm + "N"
    filenm = filenm + str(batch) + str(run) + str(reps)
    log = open(filenm + '.log', 'w+')
    print("\n──────────────────────────── TVMUI ────────────────────────────\n")
    from cpuinfo import get_cpu_info
    from datetime import datetime
    log.write("TVM Time Trial\n")
    log_print(log, "Started on " + str(datetime.now().strftime("%m/%d/%Y at %H:%M:%S")))
    log_print(log, 'Hardware: ' + supportedPlatforms[plat])
    if plat == 0:
        log_print(log, 'CPU Type: ' + get_cpu_info().get('brand_raw'))
    log_print(log, 'Backend: ' + supportedBackend[back])
    log_print(log, 'Model: ' + supportedModels[md])
    log_print(log, str(batch) + " picture(s) per run")
    log_print(log, str(run) + " run average, repeated " + str(reps) + " times.")
    if auto == 0:
        log_print(log, 'AutoTVM: Yes\n')
    else:
        log_print(log, 'AutoTVM: No\n')

    print("Loading models and images...")
    import numpy as np
    from PIL import Image
    from tvm import relay
    import tvm
    pictures = get_pics(batch)
    dataset = []
    if back == 0:
        from mxnet.gluon.model_zoo.vision import get_model
        from tvm.contrib.download import download_testdata

        if md == 0:
            model_name = "resnet18_v1"
        elif md == 1:
            model_name = "inceptionv3"
        elif md == 2:
            model_name = "mobilenetv2_1.0"
        else:
            raise Exception('Not supported!')

        block = get_model(model_name, pretrained=True)

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

        for img in pictures:
            dataset.append(transform_image(Image.open(img).resize((224, 224))))
        input_shape =[batch, 3, 224, 224]
        shape_dict = {"data": input_shape}

        mod, params = relay.frontend.from_mxnet(block, shape_dict)
        func = mod["main"]
        func = relay.Function(func.params, relay.nn.softmax(func.body), None, func.type_params, func.attrs)

    if (back == 1):
        import torch
        import torchvision
        if (modl == 0):
            model_name = "resnet18"
        if (modl == 1):
            model_name = "inceptionv3"
        if (modl == 2):
            model_name = "mobilenetv2_1.0"
        model = getattr(torchvision.models, model_name)(pretrained=True)
        model = model.eval()

        # We grab the TorchScripted model via tracing
        input_shape = [1, 3, 224, 224]
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
        shape_list = [(input_name, dataset[0].shape)]
        func, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    if (back == 2):
        import tensorflow as tf

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
        shape_dict = {"data": dataset[0].shape}
        dtype_dict = {"DecodeJpeg/contents": "uint8"}
        mod, params = relay.frontend.from_tensorflow(graph_def, layout=None, shape=shape_dict)

    target = "llvm"
    log_print(log, 'Target: ' + target)
    log_print(log, 'Actual Model: ' + model_name + '\n')
    print('Making the graph...')
    if auto == 0:
        from tvm import autotvm
        if back == 0:
            if md == 0:
                atvfile = "tunings/mxnet_Resnet_graph_opt.log"
            elif md == 2:
                atvfile = "tunings/mxnet_MobileNet_graph_opt.log"
            else:
                Exception('Not supported!')
        else:
            Exception('Not supported!')
        log_print(log, 'Using AutoTVM file ' + atvfile)
        with autotvm.apply_graph_best(atvfile):
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build(func, target, params=params)
    else:
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(func, target, params=params)

    print("\nSetting up TVM...")
    from tvm.contrib import graph_runtime
    ctx = tvm.cpu(0)
    if md == 0 or md == 2:
        dtype = "float32"
    if md == 1:
        dtype = "uint8"
    m = graph_runtime.GraphModule(lib["default"](ctx))

    def runTVM(input, number, repeat):
        arr = np.ndarray(shape=input_shape, dtype=dtype)
        i = 0
        for ip in input:
            arr[i] = ip.astype(dtype)
            i = i+1
        m.set_input("data", tvm.nd.array(arr))
        time = m.module.time_evaluator("run", ctx, number=number, repeat=repeat)()
        res = []
        if (back == 0):
            i = 0
            for i in range(len(input)):
                res.append(synset[np.argmax(m.get_output(0).asnumpy()[i])])
        if (back == 1):
            # Get top-1 result for TVM
            top1_tvm = np.argmax(m.get_output(0).asnumpy()[0])
            tvm_class_key = class_id_to_key[top1_tvm]

            # Convert input to PyTorch variable and get PyTorch result for comparison
            with torch.no_grad():
                torch_img = torch.from_numpy(input)
                output = model(torch_img)

                # Get top-1 result for PyTorch
                top1_torch = np.argmax(output.numpy())
                torch_class_key = class_id_to_key[top1_torch]
            res = key_to_classname[tvm_class_key] + "/" + key_to_classname[torch_class_key]
        if (back == 2):
            pre = np.squeeze(m.get_output(0, tvm.nd.empty(((1, 1008)), "float32")).asnumpy())
            node_lookup = tf_testing.NodeLookup(label_lookup_path=map_proto_path, uid_lookup_path=label_path)
            top_k = pre.argsort()[-5:][::-1]
            res = node_lookup.id_to_string(top_k[0])
        return [time, res]

    output = []
    total = 0
    print("\nRunning inferences...")
    for i in range(int(len(dataset)/batch)):
        log_print(log, "\nSet " + str(i) + ":")
        inp = []
        for j in range(batch):
            inp.append(dataset[batch*i + j])
        output.append(runTVM(inp, run, reps))
        e = 0
        for rl in output[i][1]:
            log_print(log, "Image " + str(e+1) + " Path: " + pictures[batch*i + e])
            log_print(log, "Image " + str(e+1) + " ID: " + rl)
            e = e+1
        log_print(log, "Time taken: " + str('%.2f' % (1000 * (output[i][0].mean))) + " ms")
        total = total + output[i][0].mean
    ave = total / int(len(dataset)/batch)
    log_print(log, '\nAVERAGE TIME: ' + str(ave * 1000) + " ms")
    log_print(log, "Finished on " + str(datetime.now().strftime("%m/%d/%Y at %H:%M:%S")))
    log.close()
    # resultMenu.show()
    return


def run_tuning():
    import os
    import numpy as np

    import tvm
    from tvm import te
    from tvm import autotvm
    from tvm import relay
    from tvm.relay import testing
    from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
    from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner
    import tvm.contrib.graph_runtime as runtime

    pat = SelectionMenu.get_selection(supportedPlatforms, "Which platform do you want to tune?",
                                      show_exit_option=False);
    model = SelectionMenu.get_selection(["Resnet", "VGG", "MobileNet", "Squeezenet", "Inception", "MXNet"],
                                        "Which model do you want to tune?", show_exit_option=False)
    if model == 5:
        submod = SelectionMenu.get_selection(supportedModels, "Which submodel do you want to tune?",
                                             show_exit_option=False)
    core = SelectionMenu.get_selection(["1", "2", "3", "4", "5", "6", "7", "8"], "How many cores do you want to use?",
                                       show_exit_option=False) + 1

    #################################################################
    # Define network
    # --------------
    # First we need to define the network in relay frontend API.
    # We can either load some pre-defined network from :code:`relay.testing`
    # or building :any:`relay.testing.resnet` with relay.
    # We can also load models from MXNet, ONNX and TensorFlow.
    #
    # In this tutorial, we choose resnet-18 as tuning example.

    def get_network(name, batch_size):
        """Get the symbol definition and random weight of a network"""
        input_shape = (batch_size, 3, 224, 224)
        output_shape = (batch_size, 1000)

        if "resnet" in name:
            n_layer = int(name.split("-")[1])
            mod, params = relay.testing.resnet.get_workload(
                num_layers=n_layer, batch_size=batch_size, dtype=dtype
            )
        elif "vgg" in name:
            n_layer = int(name.split("-")[1])
            mod, params = relay.testing.vgg.get_workload(
                num_layers=n_layer, batch_size=batch_size, dtype=dtype
            )
        elif name == "mobilenet":
            mod, params = relay.testing.mobilenet.get_workload(batch_size=batch_size, dtype=dtype)
        elif name == "squeezenet_v1.1":
            mod, params = relay.testing.squeezenet.get_workload(
                batch_size=batch_size, version="1.1", dtype=dtype
            )
        elif name == "inception_v3":
            input_shape = (1, 3, 299, 299)
            mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
        elif name == "mxnet":
            # an example for mxnet model
            from mxnet.gluon.model_zoo.vision import get_model
            if (submod == 0):
                modn = "resnet18_v1"
            if (submod == 1):
                modn = "inceptionv3"
            if (submod == 2):
                modn = "mobilenetv2_1.0"
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

    # Replace "llvm" with the correct target of your CPU.
    # For example, for AWS EC2 c5 instance with Intel Xeon
    # Platinum 8000 series, the target should be "llvm -mcpu=skylake-avx512".
    # For AWS EC2 c4 instance with Intel Xeon E5-2666 v3, it should be
    # "llvm -mcpu=core-avx2".
    target = "llvm"

    batch_size = 1
    submd = ""
    if (model == 0):
        dtype = "float32"
        model_name = "resnet-18"
    if (model == 1):
        dtype = "float32"
        model_name = "vgg-18"
    if (model == 2):
        dtype = "float32"
        model_name = "mobilenet"
    if (model == 3):
        dtype = "float32"
        model_name = "squeezenet_v1.1"
    if (model == 4):
        dtype = "float32"
        model_name = "inception_v3"
    if (model == 5):
        dtype = "float32"
        model_name = "mxnet"
        submd = supportedModels[submod]
    log_file = "logs/" + model_name + "_" + submd + ".log"
    graph_opt_sch_file = "tunings/" + model_name + "_" + submd + "_graph_opt.log"

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
            if tuner == "xgb" or tuner == "xgb-rank":
                tuner_obj = XGBTuner(task, loss_type="rank")
            elif tuner == "ga":
                tuner_obj = GATuner(task, pop_size=50)
            elif tuner == "random":
                tuner_obj = RandomTuner(task)
            elif tuner == "gridsearch":
                tuner_obj = GridSearchTuner(task)
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
        Tuner = DPTuner if use_DP else PBQPTuner
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
            ctx = tvm.cpu()
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
    run_timing()
if (go == 1):
    run_tuning()
