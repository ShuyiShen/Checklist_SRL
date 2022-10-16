from test_modules import PPAttachmentTest, LexicalTest, RobustnessTest
from test_modules import LongRangeTest, SyntaxVariationTest, ArgumentAlterTest
from test_modules import LabelConfusionTest
import nltk
nltk.download('omw-1.4')

if __name__ == "__main__":
    pp_test = PPAttachmentTest("structured-prediction-srl", save_path="saved_results/pp_test.json")
    pp_test.run_test()

#     pp_test = PPAttachmentTest("structured-prediction-srl-bert", save_path="saved_results/pp_test-bert.json")
#     pp_test.run_test()

#     lexical_test = LexicalTest("structured-prediction-srl", save_path="saved_results/lexical_test.json")
#     lexical_test.run_test()

#     lexical_test = LexicalTest("structured-prediction-srl-bert", save_path="saved_results/lexical_test-bert.json")
#     lexical_test.run_test()

#     robust_test = RobustnessTest("structured-prediction-srl", save_path="saved_results/robust_test.json")
#     robust_test.run_test()

#     robust_test = RobustnessTest("structured-prediction-srl-bert", save_path="saved_results/robust_test-bert.json")
#     robust_test.run_test()

#     long_range_test = LongRangeTest("structured-prediction-srl", save_path="saved_results/long_range_test.json")
#     long_range_test.run_test()

#     long_range_test = LongRangeTest("structured-prediction-srl-bert", save_path="saved_results/long_range_test-bert.json")
#     long_range_test.run_test()

#     syntax_test = SyntaxVariationTest("structured-prediction-srl", save_path="saved_results/syntax_vatiation_test.json")
#     syntax_test.run_test()

#     syntax_test = SyntaxVariationTest("structured-prediction-srl-bert", save_path="saved_results/syntax_vatiation_test-bert.json")
#     syntax_test.run_test()

#     arg_test = ArgumentAlterTest("structured-prediction-srl", save_path="saved_results/argument_alternation_test.json")
#     arg_test.run_test()

#     arg_test = ArgumentAlterTest("structured-prediction-srl-bert", save_path="saved_results/argument_alternation_test-bert.json")
#     arg_test.run_test()

#     label_confusion_test = LabelConfusionTest("structured-prediction-srl", save_path="saved_results/label_confusion_test.json")
#     label_confusion_test.run_test()

#     label_confusion_test = LabelConfusionTest("structured-prediction-srl-bert", save_path="saved_results/label_confusion_test-bert.json")
#     label_confusion_test.run_test()