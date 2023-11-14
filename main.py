# -*- coding: utf-8 -*-
"""
    This module is intended to join all the pipeline in separated tasks
    to be executed individually or in a flow by using command-line options

    Example:
    Dataset embedding and processing:
        $ python taskflows.py -e -pS
"""
import faulthandler

faulthandler.enable()

import argparse
import gc
import shutil
import shutil, sys, os
from argparse import ArgumentParser

from gensim.models.word2vec import Word2Vec

import configs
import src.data as data
import src.prepare as prepare
import src.process as process
import src.utils.functions.cpg as cpg
from src.utils.objects.input_dataset import InputDataset


CREATE_PARAMS = configs.Create()
PROCESS_PARAMS = configs.Process()
PATHS = configs.Paths()
FILES = configs.Files()
DEVICE = PROCESS_PARAMS.device


def select(dataset):
    result = dataset.loc[dataset["project"] == "FFmpeg"]
    len_filter = result.func.str.len() < 1200
    result = result.loc[len_filter]
    # print(len(result))
    # result = result.iloc[11001:]
    # print(len(result))
    # 暂时只使用前200条数据
    result = result.head(CREATE_PARAMS.data_size)

    return result


def setup():
    os.makedirs(PATHS.cpg, exist_ok=True)
    os.makedirs(PATHS.model, exist_ok=True)
    os.makedirs(PATHS.input, exist_ok=True)
    os.makedirs(PATHS.tokens, exist_ok=True)
    os.makedirs(PATHS.w2v, exist_ok=True)


def create_task():
    os.system(f"rm -rf {PATHS.joern} workspace")
    context = configs.Create()
    raw = data.read(PATHS.raw, FILES.raw)
    filtered = data.apply_filter(raw, select)
    filtered = data.clean(filtered)
    data.drop(filtered, ["commit_id", "project"])
    slices = data.slice_frame(filtered, context.slice_size)
    slices = [(s, slice.apply(lambda x: x)) for s, slice in slices]

    cpg_files = []
    # Create CPG binary files
    for s, slice in slices:
        data.to_files(slice, PATHS.joern)
        cpg_file = prepare.joern_parse(
            context.joern_cli_dir, PATHS.joern, PATHS.cpg, f"{s}_{FILES.cpg}"
        )
        cpg_files.append(cpg_file)
        print(f"Dataset {s} to cpg.")
        shutil.rmtree(PATHS.joern)
    # Create CPG with graphs json files
    json_files = prepare.joern_create(
        context.joern_cli_dir, PATHS.cpg, PATHS.cpg, cpg_files
    )
    for (s, slice), json_file in zip(slices, json_files):
        graphs = prepare.json_process(
            PATHS.cpg, json_file
        )  # 去除一些无用的字符串信息，并使用文件数字对每个graph进行编号
        if graphs is None:
            print(f"Dataset chunk {s} not processed.")
            continue
        dataset = data.create_with_index(
            graphs, ["Index", "cpg"]
        )  # 用上面的index对每一行数据创建索引
        dataset = data.inner_join_by_index(slice, dataset)  # 同时将分块的索引写入每一行
        print(f"Writing cpg dataset chunk {s}.")
        data.write(dataset, PATHS.cpg, f"{s}_{FILES.cpg}.pkl")
        del dataset
        gc.collect()


def embed_task():
    context = configs.Embed()
    # Tokenize source code into tokens
    dataset_files = data.get_directory_files(PATHS.cpg)  # 从data/cpg文件家中取出之前生成好的所有.pkl文件
    w2vmodel = Word2Vec(**context.w2v_args)
    w2v_init = True

    for pkl_file in dataset_files:
        file_name = pkl_file.split(".")[0]
        cpg_dataset = data.load(PATHS.cpg, pkl_file)
        tokens_dataset = data.tokenize(cpg_dataset)  # 对程序源码文本进行分词，返回分词的结果
        data.write(tokens_dataset, PATHS.tokens, f"{file_name}_{FILES.tokens}")
        # word2vec used to learn the initial embedding of each token
        w2vmodel.build_vocab(sentences=tokens_dataset.tokens, update=not w2v_init)
        w2vmodel.train(
            tokens_dataset.tokens, total_examples=w2vmodel.corpus_count, epochs=1
        )
        if w2v_init:
            w2v_init = False
        # Embed cpg to node representation and pass to graph data structure
        cpg_dataset["nodes"] = cpg_dataset.apply(
            lambda row: cpg.parse_to_nodes(row.cpg, context.max_nodes),
            axis=1,  # context.max_nodes限定了对多的节点个数，不足的补0，超过的会截断
        )
        # remove rows with no nodes
        cpg_dataset = cpg_dataset.loc[cpg_dataset.nodes.map(len) > 0]
        cpg_dataset["input"] = cpg_dataset.apply(
            lambda row: prepare.nodes_to_input(
                row.nodes, row.target, context.max_nodes, w2vmodel.wv, context.edge_type
            ),
            axis=1,
        )  # 使用w2vec对每一个节点的文本进行编码
        data.drop(cpg_dataset, ["nodes"])
        print(f"Saving input dataset {file_name} with size {len(cpg_dataset)}.")
        data.write(
            cpg_dataset[["input", "target"]], PATHS.input, f"{file_name}_{FILES.input}"
        )
        del cpg_dataset
        gc.collect()
    print("Saving w2vmodel.")
    w2vmodel.save(f"{PATHS.w2v}/{FILES.w2v}")


def process_task(stopping, test_only=False):
    context = configs.Process()
    devign = configs.Devign()
    model_path = PATHS.model + FILES.model

    model = process.Devign(
        path=model_path,
        device=DEVICE,
        model=devign.model,
        max_nodes=configs.Embed().max_nodes,
        learning_rate=devign.learning_rate,
        weight_decay=devign.weight_decay,
        loss_lambda=devign.loss_lambda,
    )
    train = process.Train(model, context.epochs)
    input_dataset = data.loads(PATHS.input, PROCESS_PARAMS.dataset_ratio)
    # split the dataset and pass to DataLoader with batch size
    train_loader, val_loader, test_loader = list(
        map(
            lambda x: x.get_loader(context.batch_size, True),
            data.train_val_test_split(input_dataset, False),
        )
    )
    train_loader_step = process.LoaderStep("Train", train_loader, DEVICE)
    val_loader_step = process.LoaderStep("Validation", val_loader, DEVICE)
    test_loader_step = process.LoaderStep("Test", test_loader, DEVICE)

    if test_only:
        model.load()
        process.predict(model, test_loader_step)
        return

    print(f"train with {DEVICE}.")
    if stopping:
        early_stopping = process.EarlyStopping(
            model,
            patience=context.patience,
            verbose=PROCESS_PARAMS.verbose,
            delta=PROCESS_PARAMS.delta,
        )
        train(train_loader_step, val_loader_step, early_stopping)
        model.load()
    else:
        train(train_loader_step, val_loader_step)
        model.save()

    process.predict(model, test_loader_step)


def main():
    """
    main function that executes tasks based on command-line options
    """
    parser: ArgumentParser = argparse.ArgumentParser()
    # parser.add_argument('-p', '--prepare', help='Prepare task', required=False)
    parser.add_argument("-c", "--create", action="store_true")
    parser.add_argument("-e", "--embed", action="store_true")
    parser.add_argument("-p", "--process", action="store_true")
    parser.add_argument("-t", "--test", action="store_true")
    parser.add_argument("-pS", "--process_stopping", action="store_true")

    args = parser.parse_args()

    setup()
    if args.create:
        create_task()
    if args.embed:
        embed_task()
    if args.process:
        process_task(False)
    if args.process_stopping:
        process_task(True)
    if args.test:
        process_task(False, True)


if __name__ == "__main__":
    # main()
    print(DEVICE)
