from datasets import load_dataset

def prepdata():
        data_mc: list[dict[str, str]] = []
        data_tf: list[dict[str, str]] = []
        #
        data_train: dict[str, list[str]] = {"text": [], "question":[], "label":[]}
        data_test: dict[str, list[str]] = {"text": [], "question":[], "label":[]}
        data_eval: dict[str, list[str]] = {"text": [], "question": [], "label": []}
        #
        dataset = load_dataset("agentic-learning-ai-lab/daily-oracle", "default")
        #
        r: int = 0
        #
        n_mc: int = 0
        n_tf: int = 0
        #
        strange: list[str] = []
        #
        for i, row in enumerate(dataset["train"]):  # type: ignore
            #
            if row["date"] < "2024":  # type: ignore
                #
                continue
            #
            if "".join([l for l in row["answer"].lower() if l >= "a" and l <= "z" ]) in ["a", "b", "c", "d"]:
                #
                n_mc += 1
                #
                label: str = ""
                #
                if "a" in row["answer"].lower(): label = "A"
                elif "b" in row["answer"].lower(): label = "B"
                elif "c" in row["answer"].lower(): label = "C"
                elif "d" in row["answer"].lower(): label = "D"
                #
                if label == "":
                    #
                    continue
                #
                data_mc.append(
                    {
                        "question": f"Question: {row['question']} Choices: A: {row['choice_a']}; B: {row['choice_b']}; C: {row['choice_c']}; D: {row['choice_d']}. Answer with only one letter A, B, C or D. Answer: ",
                        "label": label,
                        "title": row["title"],
                        "text": row["text"],
                        "date": row["date"]
                    }
                )
            #
            elif "".join([l for l in row["answer"].lower() if l >= "a" and l <= "z" ]) in ["yes", "no"]:
                continue
                #
                n_tf += 1
                #
                label: str = ""
                #
                if "yes" in row["answer"].lower(): label = "yes"
                elif "no" in row["answer"].lower(): label = "no"
                #
                if label == "":
                    #
                    continue
                #
                data_tf.append(
                    {
                        "question": f"Question:\n\n{row['question']}\n\nIs it true or false?\n\nAnswer only one word true or false.\n\nAnswer: ",
                        "label": label,
                        "title": row["title"],
                        "text": row["text"],
                        "date": row["date"]
                    }
                )
            #
            else:
                #
                continue
            #
            data_mc.sort(key=lambda x: x["date"])
            data_tf.sort(key=lambda x: x["date"])
            #
            if False:
                #
                print(f"\n\nrow {i} ({r}):")
                #
                for k, v in row.items():  # type: ignore
                    #
                    print(f"\t- `{k}` => {type(v)}", end="")  # type: ignore
                    #
                    if hasattr(v, "__len__"):  # type: ignore
                        #
                        print(f" | {len(v)}", end="")  # type: ignore
                        #
                        if len(v) <= 20:  # type: ignore
                            #
                            print(f" -> value = `{v}`")
                    else:
                        #
                        print(f" -> value = {v}")
                    #
                    print()
                #
                print()
            #
            r += 1
        #
        nb_eval: int = 1000
        nb_test: int = 1000
        nb_train_mc: int = n_mc - nb_eval - nb_test
        nb_train_tf: int = n_tf - nb_eval - nb_test
        #
        cursor_mc: int = 0
        cursor_tf: int = 0
        #
        while cursor_mc < n_mc or cursor_tf < n_tf:
            #
            next_type: str = "mc"
            if cursor_mc >= n_mc:
                #
                next_type = "tf"
            #
            elif cursor_tf >= n_tf:
                #
                next_type = "mc"
            #
            elif data_tf[cursor_tf]["date"] > data_mc[cursor_mc]["date"]:
                #
                next_type = "tf"
            #
            elt_to_add: dict[str, Any]
            where_to_add: str = "train"
            #
            if next_type == "mc":
                #
                elt_to_add = data_mc[cursor_mc]
                #
                if cursor_mc >= nb_train_mc:
                    #
                    where_to_add = "test"
                #
                if cursor_mc >= nb_train_mc + nb_test:
                    #
                    where_to_add = "eval"
                #
                cursor_mc += 1
            #
            else:
                #
                elt_to_add = data_tf[cursor_tf]
                #
                if cursor_tf >= nb_train_tf:
                    #
                    where_to_add = "test"
                #
                if cursor_tf >= nb_train_tf + nb_test:
                    #
                    where_to_add = "eval"
                #
                cursor_tf += 1
                #
            if where_to_add == "train":
                data_train["text"].append( elt_to_add["text"] )
                data_train["question"].append( elt_to_add["question"] )
                data_train["label"].append( elt_to_add["label"] )
            elif where_to_add == "test":
                data_test["text"].append( elt_to_add["text"] )
                data_test["question"].append( elt_to_add["question"] )
                data_test["label"].append( elt_to_add["label"] )
            else:
                data_eval["text"].append( elt_to_add["text"] )
                data_eval["question"].append( elt_to_add["question"] )
                data_eval["label"].append( elt_to_add["label"] )

        #
        print(f"DEBUG | strange = {strange}")
        #
        print(f"\nN_MC = {n_mc} | N_TF = {n_tf} | TOT = {r}")
        #
        print(f"DEBUG | len(data_train) = {len(data_train["text"])} | len(data_test) = {len(data_test["text"])} | len(data_eval) = {len(data_eval["question"])}")
        #
        # train_dataset: Dataset = Dataset.from_dict(data_train)  # type: ignore
        # test_dataset: Dataset = Dataset.from_dict(data_test)  # type: ignore
        # validation_dataset: Dataset = Dataset.from_dict(data_eval)  # type: ignore

        n_text = 5

        txttrain=""
        txttest=""
        for i in range(len(data_train["question"])-1000, len(data_train["question"])):
            txttrain += " ".join([data_train["text"][j].replace('\n',' ') for j in range(max(0, i - n_text), i+1)])+'. '
            q = data_train["question"][i]
            l = data_train["label"][i]
            txttrain += q.replace('\n',' ') + l + "\n"
        for i in range(len(data_test["question"])):
            txttest += " ".join([data_test["text"][j].replace('\n',' ') for j in range(max(0, i - n_text), i+1)])+'. '
            q = data_test["question"][i]
            l = data_test["label"][i]
            txttest += q.replace('\n',' ') + l + "\n"

        while "  " in txttrain: txttrain = txttrain.replace("  ", " ")
        while "  " in txttest: txttest = txttest.replace("  ", " ")

        with open("fntrain.txt","w",encoding="utf8") as f:
            f.write(txttrain)
        with open("fntest.txt","w",encoding="utf8") as f:
            f.write(txttest)

prepdata()

