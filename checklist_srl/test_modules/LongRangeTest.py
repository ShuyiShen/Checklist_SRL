import json
import checklist
from pprint import pprint
from checklist.editor import Editor
from checklist.test_types import MFT
from checklist.expect import Expect
from checklist.pred_wrapper import PredictorWrapper
from allennlp_models.pretrained import load_predictor
import nltk
nltk.download('omw-1.4')

class LongRangeTest(object):
    def __init__(self, model_name, save_path=None):
        """
        :param model_name: model name used in allennlp_models
        :param save_path: where to save test result, if None : not to save
        """
        self.model_name = model_name
        self.model = load_predictor(model_name)
        self.save_path = save_path
        self.predict_wrapper = PredictorWrapper.wrap_predict(
            lambda data: [self.model.predict(item) for item in data]
        )
        self.editor = Editor()

    def format_srl(self, x, pred, conf, label=None, meta=None):
        predicate_structure = pred['verbs'][0]['description']

        return predicate_structure

    def expect_func(self, x, pred, conf, label=None, meta=None):
        # people should be recognized as arg1

        people = "The " + meta['mask'][0]
        obj = "a good " + meta['mask'][1]

        arg_0 = " ".join(get_arg(pred, arg_target='ARG0'))
        arg_1 = " ".join(get_arg(pred, arg_target='ARG1'))

        if arg_0 == people and arg_1 == obj:
            pass_ = True
        else:
            pass_ = False
        return pass_

    def construct_template(self):
        template_list = []

        t1 = self.editor.template(
            "The {mask}, though in a depression,  actually offered a good {mask}.", meta=True, nsamples=10
        )
        template_list.append(t1)
        return template_list

    def run_test(self):
        samples = []
        metas = []
        prediction = []
        for t in self.construct_template():
            test = MFT(**t, expect=Expect.single(self.expect_func))
            test.run(self.predict_wrapper)
            test.summary(format_example_fn=self.format_srl)
            samples.extend(t.data)
            metas.extend(t.meta)
            prediction.extend(test.results["preds"])

        if self.save_path:
            self.save_results(samples=samples, metas=metas, predictions=prediction)

    def save_results(self, samples, metas, predictions):
        results = [{"Test": type(self).__name__, "model": self.model_name}]
        for sample, meta, pred in zip(samples, metas, predictions):
            item = {"sentence": sample}
            target = "ARG0 and ARG1"
            expected = ["The " + meta["mask"][0], "a good " + meta["mask"][1]]
            predict = [" ".join(list(get_arg(pred, "ARG0"))), " ".join(list(get_arg(pred, "ARG1")))]
            item["target"] = target
            item["expected"] = expected
            item["prediction"] = predict
            results.append(item)

        with open(self.save_path, "w", encoding="utf-8") as fw:
            for item in results:
                fw.write(json.dumps(item, ensure_ascii=False) + "\n")
        print("saved results!")


def get_arg(pred, arg_target='ARG1'):
    # we assume one predicate:
    predicate_arguments = pred['verbs'][0]
    words = pred['words']
    tags = predicate_arguments['tags']

    arg_list = []
    for t, w in zip(tags, words):
        arg = t
        if '-' in t:
            arg = t.split('-')[1]
        if arg == arg_target:
            arg_list.append(w)
    # arg_set = set(arg_list)
    # return arg_set
    return arg_list
