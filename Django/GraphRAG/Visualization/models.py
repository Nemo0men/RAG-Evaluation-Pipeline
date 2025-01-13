from django.db import models

class Article(models.Model):
    title = models.CharField(max_length=255)

    def __str__(self):
        return self.title

class Paragraph(models.Model):
    article = models.ForeignKey(Article, on_delete=models.CASCADE, related_name='paragraphs')
    context = models.TextField()

    def __str__(self):
        return f"Paragraph in '{self.article.title}'"

class QuestionAnswer(models.Model):
    paragraph = models.ForeignKey(Paragraph, on_delete=models.CASCADE, related_name='qas')
    question = models.TextField()
    answer = models.TextField()
    answer_start = models.IntegerField()
    is_impossible = models.BooleanField(default=False)

    def __str__(self):
        return f"Q: {self.question} (Ans: {self.answer})"