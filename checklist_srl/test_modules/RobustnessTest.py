import json
import spacy
import checklist
from pprint import pprint
from checklist.editor import Editor
from checklist.test_types import MFT, INV
from checklist.expect import Expect
from checklist.pred_wrapper import PredictorWrapper
from checklist.perturb import Perturb
from allennlp_models.pretrained import load_predictor
import nltk
nltk.download('omw-1.4')

nlp = spacy.load("en_core_web_sm")





class RobustnessTest(object):
    def __init__(self, model_name, save_path=None):
        """
        :param model_name: model name used in allennlp_models
        :param save_path: where to save test result, if None : not to save
        """
        self.model_name = model_name
        self.model = load_predictor(model_name)
        self.save_path = save_path
        # self.predict_wrapper = PredictorWrapper.wrap_predict(
        #     lambda data: [self.model.predict(item)['verbs'][0]['tags'] for item in data if self.model.predict(item)['verbs'] else []]
        # )
        self.predict_wrapper = PredictorWrapper.wrap_predict(self.predict_srl)
        self.editor = Editor()

    def predict_srl(self, data):

        pred = []
        for d in data:
            p = self.model.predict(d)['verbs']
            if p:
                pred.append(p[0]['tags'])
            else:
                pred.append([])
        return pred

    def format_srl(self, x, pred, conf, label=None, meta=None):
        # predicate_structure = pred['verbs'][0]['description']
        # predicate_structure = pred['verbs'][0]['tags']

        return pred

    def expect_func(self, x, pred, conf, label=None, meta=None):
        # people should be recognized as arg1

        people = {meta['first_name'], meta['last_name']}
        arg_1 = get_arg(pred, arg_target='ARG1')

        if arg_1 == people:
            pass_ = True
        else:
            pass_ = False
        return pass_

    def construct_template(self):
        template_list = []
        data = [
            "The boy broke the vase.",
            "John sold Tom the car.",
            "Jack sprayed paint on the wall.",
            "The mushrooms were sliced by the cook.",
            "The river froze solid."
        ]
        t1 = Perturb.perturb(data, Perturb.add_typos)
        template_list.append(t1)
        # data = list(nlp.pipe(data))
        # t2 = Perturb.perturb(data, Perturb.contractions)
        # template_list.append(t2)

        return template_list

    def run_test(self):
        samples = []
        prediction = []
        for t in self.construct_template():
            print(t)
            test = INV(**t)
            test.run(self.predict_wrapper)
            # test.summary()
            test.summary(format_example_fn=self.format_srl)
            # print(test.results)
            samples.extend(t.data)
            prediction.extend(test.results["preds"])

        if self.save_path:
            self.save_results(samples=samples, predictions=prediction)

    def save_results(self, samples, predictions):
        results = [{"Test": type(self).__name__, "model": self.model_name}]
        for sample, pred in zip(samples, predictions):
            item = {"sentence": sample}
            target = "INV"
            expected = list(pred[0])
            predict = list(pred[1])
            # print(expected)
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
    arg_set = set(arg_list)
    return arg_set
