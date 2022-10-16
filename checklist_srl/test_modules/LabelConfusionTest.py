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

class LabelConfusionTest(object):
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

        arg_word = meta['mask'][1]
        arg_role = get_label(pred, arg_word)

        if arg_role == "ARG2":
            pass_ = True
        else:
            pass_ = False
        return pass_

    def construct_template(self):
        template_list = []

        t1 = self.editor.template(
            "The worker move the {mask} into that building with a {mask}.", meta=True, nsamples=10
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
            target = "with a " + meta['mask'][1]
            expected = "ARG2"
            predict = get_label(pred, meta['mask'][1])
            item["target"] = target
            item["expected"] = expected
            item["prediction"] = predict
            results.append(item)

        with open(self.save_path, "w", encoding="utf-8") as fw:
            for item in results:
                fw.write(json.dumps(item, ensure_ascii=False) + "\n")
        print("saved results!")


def get_label(pred, arg_word=None):
    # we assume one predicate:
    predicate_arguments = pred['verbs'][0]
    words = pred['words']
    tags = predicate_arguments['tags']
    idx = words.index(arg_word)
    arg_role = tags[idx][tags[idx].find("-")+1:]
    return arg_role
