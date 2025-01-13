import json
from django.core.management.base import BaseCommand
from ...models import Article, Paragraph, QuestionAnswer

class Command(BaseCommand):
    help = "Load SQuAD dataset into the database"

    def add_arguments(self, parser):
        parser.add_argument('file_path', type=str, help='Path to the SQuAD JSON file')

    def handle(self, *args, **kwargs):
        file_path = kwargs['file_path']
        with open(file_path, 'r') as f:
            data = json.load(f)

        for article_data in data['data']:
            article = Article.objects.create(title=article_data['title'])

            for paragraph_data in article_data['paragraphs']:
                paragraph = Paragraph.objects.create(
                    article=article,
                    context=paragraph_data['context']
                )

                for qa in paragraph_data['qas']:
                    QuestionAnswer.objects.create(
                        paragraph=paragraph,
                        question=qa['question'],
                        answer=qa['answers'][0]['text'] if qa['answers'] else '',
                        answer_start=qa['answers'][0]['answer_start'] if qa['answers'] else -1,
                        is_impossible=qa['is_impossible']
                    )

        self.stdout.write(self.style.SUCCESS("SQuAD data successfully loaded"))
