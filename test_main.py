import logging
from unittest import TestCase

from transformers import MarianTokenizer, MarianMTModel

import main

logger = logging.getLogger(__name__)


class TestToken(TestCase):
    def test_text(self):
        my_token = main.Token()
        my_token.text = "test"
        self.assertEqual(my_token.text, "test")
        del my_token
        my_token = main.Token(text="test")
        self.assertEqual(my_token.text, "test")

    def test_is_placeholder(self):
        my_token = main.Token()
        my_token.is_placeholder = True
        self.assertTrue(my_token.is_placeholder)
        del my_token
        my_token = main.Token(is_placeholder=True)
        self.assertTrue(my_token.is_placeholder)

    def test__eq__(self):
        my_token_1 = main.Token("this", is_placeholder=False)
        my_token_eq = main.Token("this", is_placeholder=False)
        my_token_diff_1 = main.Token("another", is_placeholder=False)
        my_token_diff_2 = main.Token("this", is_placeholder=True)
        self.assertEqual(my_token_1, my_token_eq)
        self.assertNotEqual(my_token_1, my_token_diff_1)
        self.assertNotEqual(my_token_1, my_token_diff_2)


class TestReplacement(TestCase):
    def test_entity_name(self):
        my_replacement = main.Replacement()
        my_replacement.entity_name = "test_entity_name"
        self.assertEqual(my_replacement.entity_name, "test_entity_name")
        del my_replacement
        my_replacement = main.Replacement(entity_name="test_entity_name")
        self.assertEqual(my_replacement.entity_name, "test_entity_name")

    def test_entity_value(self):
        my_replacement = main.Replacement()
        my_replacement.entity_value = "test value"
        self.assertEqual(my_replacement.entity_value, "test value")
        del my_replacement
        my_replacement = main.Replacement(entity_value="test value")
        self.assertEqual(my_replacement.entity_value, "test value")

    def test__eq__(self):
        my_replacement_1 = main.Replacement("test_eq", "val")
        my_replacement_eq = main.Replacement("test_eq", "val")
        my_replacement_diff_1 = main.Replacement("test_diff", "val")
        my_replacement_diff_2 = main.Replacement("test_eq", "diff")
        self.assertEqual(my_replacement_1, my_replacement_eq)
        self.assertNotEqual(my_replacement_1, my_replacement_diff_1)
        self.assertNotEqual(my_replacement_1, my_replacement_diff_2)


class TestReplacementSet(TestCase):
    def test_replacements(self):
        my_rep_set = main.ReplacementSet()
        my_rep_set.replacements = [
            main.Replacement("attribute", "stupid"),
            main.Replacement("count", "2"),
        ]
        self.assertEqual(
            my_rep_set.replacements,
            [main.Replacement("attribute", "stupid"), main.Replacement("count", "2")],
        )
        del my_rep_set
        my_rep_set = main.ReplacementSet({"attribute": "wrong", "count": "3"})
        self.assertEqual(
            my_rep_set.replacements,
            [main.Replacement("attribute", "wrong"), main.Replacement("count", "3")],
        )

    def test__eq__(self):
        replacement_set_1 = main.ReplacementSet(
            replacements={"slot1": "value1", "slot2": "value2"}
        )
        replacement_set_2 = main.ReplacementSet(
            replacements={"slot1": "value1", "slot2": "value2"}
        )
        replacement_set_3 = main.ReplacementSet(
            replacements={"slot2": "value2", "slot1": "value1"}
        )
        replacement_set_diff = main.ReplacementSet(replacements={"slot1": "value1"})
        self.assertEqual(replacement_set_1, replacement_set_2)
        self.assertEqual(replacement_set_1, replacement_set_3)
        self.assertNotEqual(replacement_set_1, replacement_set_diff)

    def test_parse_dict(self):
        my_rep_set = main.ReplacementSet()
        dict_replacements = {"attribute": "stupid", "count": "2"}
        my_rep_set.parse_dict(dict_replacements)
        self.assertEqual(
            my_rep_set.replacements,
            [main.Replacement("attribute", "stupid"), main.Replacement("count", "2")],
        )


class TestTokenizer(TestCase):
    def test_tokenize_sequence(self):
        template_string = "this is a {attribute} template with {count} slots"
        my_tokenizer = main.Tokenizer()
        self.assertEqual(
            my_tokenizer.tokenize_sequence(template_string),
            [
                main.Token(text="this", is_placeholder=False),
                main.Token(text="is", is_placeholder=False),
                main.Token(text="a", is_placeholder=False),
                main.Token(text="attribute", is_placeholder=True),
                main.Token(text="template", is_placeholder=False),
                main.Token(text="with", is_placeholder=False),
                main.Token(text="count", is_placeholder=True),
                main.Token(text="slots", is_placeholder=False),
            ],
        )


class TestTokenSequence(TestCase):
    token_list_2 = [
        main.Token(text="this", is_placeholder=False),
        main.Token(text="is", is_placeholder=False),
        main.Token(text="a", is_placeholder=False),
        main.Token(text="value", is_placeholder=True),
    ]

    def test_tokens(self):
        my_template = main.TokenSequence()
        my_template.tokens = self.token_list_2
        self.assertEqual(my_template.tokens, self.token_list_2)
        del my_template
        my_template = main.TokenSequence(self.token_list_2)
        self.assertEqual(my_template.tokens, self.token_list_2)

    def test__eq__(self):
        sequence_1 = main.TokenSequence(
            [
                main.Token(text="word1", is_placeholder=False),
                main.Token(text="word2", is_placeholder=False),
                main.Token(text="word3", is_placeholder=False),
            ]
        )
        sequence_2 = main.TokenSequence(
            [
                main.Token(text="word1", is_placeholder=False),
                main.Token(text="word2", is_placeholder=False),
                main.Token(text="word3", is_placeholder=False),
            ]
        )
        sequence_3 = main.TokenSequence(
            [
                main.Token(text="word1", is_placeholder=False),
                main.Token(text="word3", is_placeholder=False),
                main.Token(text="word2", is_placeholder=False),
            ]
        )
        sequence_4 = main.TokenSequence(
            [
                main.Token(text="word1", is_placeholder=False),
                main.Token(text="word2", is_placeholder=False),
            ]
        )

        self.assertEqual(sequence_1, sequence_2)
        self.assertNotEqual(sequence_1, sequence_3)
        self.assertNotEqual(sequence_1, sequence_4)

    my_tokenizer = main.Tokenizer()

    def test_from_string(self):
        my_template = main.TokenSequence()
        template_string = "this is a {attribute} template with {count} slots"
        my_template.from_string(template_string, self.my_tokenizer)
        self.assertEqual(
            my_template.tokens,
            [
                main.Token(text="this", is_placeholder=False),
                main.Token(text="is", is_placeholder=False),
                main.Token(text="a", is_placeholder=False),
                main.Token(text="attribute", is_placeholder=True),
                main.Token(text="template", is_placeholder=False),
                main.Token(text="with", is_placeholder=False),
                main.Token(text="count", is_placeholder=True),
                main.Token(text="slots", is_placeholder=False),
            ],
        )

    def test_as_string(self):
        my_token_seq = main.TokenSequence(
            [
                main.Token(text="this", is_placeholder=False),
                main.Token(text="is", is_placeholder=False),
                main.Token(text="a", is_placeholder=False),
                main.Token(text="test", is_placeholder=False),
            ]
        )
        self.assertEqual(my_token_seq.as_string(self.my_tokenizer), "this is a test")
        my_token_seq = main.TokenSequence(
            [
                main.Token(text="this", is_placeholder=False),
                main.Token(text="attribute", is_placeholder=True),
                main.Token(text="is", is_placeholder=False),
                main.Token(text="value", is_placeholder=True),
            ]
        )
        self.assertEqual(
            my_token_seq.as_string(self.my_tokenizer), "this {attribute} is {value}"
        )

    def test_lexicalize(self):
        my_template = main.TokenSequence()
        template_string = "this is a {attribute} template with {count} slots"
        my_template.from_string(template_string, self.my_tokenizer)
        my_rep_set = main.ReplacementSet()
        dict_replacements = {"attribute": "stupid", "count": "2"}
        my_rep_set.parse_dict(dict_replacements)
        out = my_template.lexicalize(my_rep_set)
        self.assertEqual(
            out,
            [
                main.Token(text="this", is_placeholder=False),
                main.Token(text="is", is_placeholder=False),
                main.Token(text="a", is_placeholder=False),
                main.Token(text="stupid", is_placeholder=False),
                main.Token(text="template", is_placeholder=False),
                main.Token(text="with", is_placeholder=False),
                main.Token(text="2", is_placeholder=False),
                main.Token(text="slots", is_placeholder=False),
            ],
        )

    full_seq = main.TokenSequence(
        [
            main.Token(text="word1", is_placeholder=False),
            main.Token(text="word2", is_placeholder=False),
            main.Token(text="word3", is_placeholder=False),
            main.Token(text="word4", is_placeholder=False),
        ]
    )

    start_seq = main.TokenSequence(
        [
            main.Token(text="word1", is_placeholder=False),
            main.Token(text="word2", is_placeholder=False),
        ]
    )

    mid_seq = main.TokenSequence(
        [
            main.Token(text="word2", is_placeholder=False),
            main.Token(text="word3", is_placeholder=False),
        ]
    )

    other_seq = main.TokenSequence(
        [
            main.Token(text="test", is_placeholder=False),
            main.Token(text="sequence", is_placeholder=False),
        ]
    )

    def test_startswith(self):
        self.assertTrue(self.full_seq.startswith(self.start_seq))
        self.assertFalse(self.full_seq.startswith(self.mid_seq))

    def test_match_subtokens(self):
        self.assertTrue(self.full_seq.match_subtokens(self.mid_seq))
        self.assertFalse(self.full_seq.match_subtokens(self.other_seq))
        longer_seq = main.TokenSequence(
            [
                main.Token(text="another", is_placeholder=False),
                main.Token(text="very", is_placeholder=False),
                main.Token(text="long", is_placeholder=False),
                main.Token(text="test", is_placeholder=False),
                main.Token(text="sequence", is_placeholder=False),
            ]
        )
        self.assertFalse(self.full_seq.match_subtokens(longer_seq))

    def test_match_subtokens_with_index(self):
        self.assertEqual(self.full_seq.match_subtokens_with_index(self.mid_seq), 1)
        self.assertEqual(self.full_seq.match_subtokens_with_index(self.other_seq), -1)


class TestTemplate(TestCase):
    def test_get_realizations(self):
        my_template = main.Template(
            tokens=main.TokenSequence(
                [
                    main.Token(text="entity", is_placeholder=True),
                    main.Token(text="is", is_placeholder=False),
                    main.Token(text="attribute", is_placeholder=True),
                ]
            ),
            replacement_sets=[
                main.ReplacementSet({"entity": "sky", "attribute": "blue"}),
                main.ReplacementSet({"entity": "snow", "attribute": "white"}),
            ],
        )
        self.assertEqual(
            my_template.get_realizations(),
            [
                main.TokenSequence(
                    [
                        main.Token("sky", False),
                        main.Token("is", False),
                        main.Token("blue", False),
                    ]
                ),
                main.TokenSequence(
                    [
                        main.Token("snow", False),
                        main.Token("is", False),
                        main.Token("white", False),
                    ]
                ),
            ],
        )
        with self.assertRaises(ValueError):
            my_template.get_realizations(as_text=True)
        tokenizer = main.Tokenizer()
        self.assertEqual(
            my_template.get_realizations(as_text=True, tokenizer=tokenizer),
            ["sky is blue", "snow is white"],
        )


class TestTranslator(TestCase):
    model_name = "Helsinki-NLP/opus-mt-en-it"
    mt_tokenizer = MarianTokenizer.from_pretrained(model_name)
    mt_model = MarianMTModel.from_pretrained(model_name)

    translator = main.Translator(tokenizer=mt_tokenizer, model=mt_model)

    def test_translate(self):
        logger.debug("Testing with input type List[str]")
        translation = self.translator.translate(["this", "is", "easy"])
        translation = [t.lower() for t in translation]
        self.assertEqual(translation, ["questo", "è", "facile"])

        logger.debug("Testing with input type Replacement")
        replacement = main.Replacement(entity_name="slot1", entity_value="easy")
        translation = self.translator.translate(replacement)
        self.assertEqual(
            translation, main.Replacement(entity_name="slot1", entity_value="facile")
        )

        logger.debug("Testing with input type Replacement, entity_values with multiple tokens")
        replacement = main.Replacement(entity_name="slot1", entity_value="to be")
        translation = self.translator.translate(replacement)
        self.assertEqual(
            translation, main.Replacement(entity_name="slot1", entity_value="essere")
        )
        replacement = main.Replacement(entity_name="slot1", entity_value="office desk")
        translation = self.translator.translate(replacement)
        self.assertEqual(
            translation, main.Replacement(entity_name="slot1", entity_value="scrivania per ufficio")
        )

        logger.debug("Testing with input type ReplacementSet")
        replacement_set = main.ReplacementSet(
            replacements={"slot1": "is", "slot2": "easy",}
        )
        translation = self.translator.translate(replacement_set)
        self.assertEqual(
            translation,
            main.ReplacementSet(replacements={"slot1": "è", "slot2": "facile"}),
        )
        logger.debug("Testing with input type ReplacementSet, entity_values with multiple tokens")
        replacement_set = main.ReplacementSet(
            replacements={"slot1": "to be", "slot2": "office desk",}
        )
        translation = self.translator.translate(replacement_set)
        self.assertEqual(
            translation,
            main.ReplacementSet(replacements={"slot1": "essere", "slot2": "scrivania per ufficio"}),
        )

        logger.debug("Testing with input type TokenSequence")
        source = main.TokenSequence(
            [
                main.Token(text="entity", is_placeholder=True),
                main.Token(text="is", is_placeholder=False),
                main.Token(text="easy", is_placeholder=False),
            ]
        )
        expected_out = main.TokenSequence(
            [
                main.Token(text="entity", is_placeholder=True),
                main.Token(text="è", is_placeholder=False),
                main.Token(text="facile", is_placeholder=False),
            ]
        )
        translation = self.translator.translate(source)
        logger.debug(translation)
        self.assertEqual(self.translator.translate(source), expected_out)

        logger.debug("Testing with input type TokenSequence with words that get translated to multiple tokens")
        source = main.TokenSequence(
            [
                main.Token(text="quantity", is_placeholder=True),
                main.Token(text="snowballs", is_placeholder=False),
            ]
        )
        expected_out = main.TokenSequence(
            [
                main.Token(text="quantity", is_placeholder=True),
                main.Token(text="palle di neve", is_placeholder=False),
            ]
        )
        translation = self.translator.translate(source)
        logger.debug(translation)
        self.assertEqual(self.translator.translate(source), expected_out)

        logger.debug("Testing with input type Template")
        template_frame = main.TokenSequence()
        translated_frame = main.TokenSequence()
        tokenizer = main.Tokenizer()
        template_frame.from_string("{object} is {attribute}", tokenizer)
        translated_frame.from_string("il {object} è {attribute}", tokenizer)
        replacement_sets = [
            main.ReplacementSet({"object": "sky", "attribute": "blue"}),
        ]
        translated_repl_sets = [
            main.ReplacementSet({"object": "cielo", "attribute": "blu"}),
        ]
        my_template = main.Template(
            tokens=template_frame, replacement_sets=replacement_sets
        )
        translation = self.translator.translate(my_template)
        self.assertEqual(translation[0].tokens, translated_frame)
        self.assertEqual(translation[0].replacement_sets, translated_repl_sets)

        logger.debug("Testing with input type Template with entity values that get translated to multiple tokens")
        template_frame = main.TokenSequence()
        translated_frame = main.TokenSequence()
        tokenizer = main.Tokenizer()
        template_frame.from_string(" the {object} is {attribute}", tokenizer)
        translated_frame.from_string("la {object} è {attribute}", tokenizer)
        replacement_sets = [
            main.ReplacementSet({"object": "mailbox", "attribute": "big"}),
        ]
        translated_repl_sets = [
            main.ReplacementSet({"object": "casella di posta", "attribute": "grande"}),
        ]
        my_template = main.Template(
            tokens=template_frame, replacement_sets=replacement_sets
        )
        translation = self.translator.translate(my_template)
        self.assertEqual(translation[0].tokens, translated_frame)
        self.assertEqual(translation[0].replacement_sets, translated_repl_sets)

        logger.debug("Testing with input type Template with entity values that get translated to multiple tokens and multi-token entity values")
        template_frame = main.TokenSequence()
        translated_frame = main.TokenSequence()
        tokenizer = main.Tokenizer()
        template_frame.from_string(" the {object} is {attribute}", tokenizer)
        translated_frame.from_string("la {object} è {attribute}", tokenizer)
        replacement_sets = [
            main.ReplacementSet({"object": "mailbox", "attribute": "very big"}),
        ]
        translated_repl_sets = [
            main.ReplacementSet({"object": "casella di posta", "attribute": "molto grande"}),
        ]
        my_template = main.Template(
            tokens=template_frame, replacement_sets=replacement_sets
        )
        translation = self.translator.translate(my_template)
        self.assertEqual(translation[0].tokens, translated_frame)
        self.assertEqual(translation[0].replacement_sets, translated_repl_sets)

        # TODO: add more tests (e.g. with multiple repl sets)

        logger.debug("Testing with unsupported input type")
        with self.assertRaises(TypeError):
            self.translator.translate(0.5)
