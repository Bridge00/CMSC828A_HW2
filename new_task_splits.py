from io_utils import read_json, write_json
import numpy as np

if __name__ == '__main__':
    task_split_meta = read_json("./task_splits.json")

    task_to_class_list = task_split_meta["task_to_class_list"]

    new_task_to_class = dict()
    task_lists = [str(i + 1) for i in range(5)]
    for task in task_lists:
        new_lists = []
        for new_task in task_lists:
            if new_task == task:
                continue
            new_lists += [int(i) for i in np.random.choice(task_to_class_list[new_task], size=5).astype(np.int32)]
        new_task_to_class[task] = new_lists + task_to_class_list[task]
    task_split_meta["task_to_class_list"] = new_task_to_class
    write_json(task_split_meta, "./new_task_splits.json")