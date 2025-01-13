from django.shortcuts import render
from django.db.models import Count, Avg
from .models import Article, Paragraph, QuestionAnswer
from django.http import HttpResponse

def dashboard_view(request):
    # Metrics
    total_articles = Article.objects.count()
    total_paragraphs = Paragraph.objects.count()
    total_questions = QuestionAnswer.objects.count()
    impossible_questions = QuestionAnswer.objects.filter(is_impossible=True).count()

    # Context for template
    context = {
        'total_articles': total_articles,
        'total_paragraphs': total_paragraphs,
        'total_questions': total_questions,
        'impossible_questions': impossible_questions,
    }
    return render(request, 'dashboard.html', context)

def home_view(request):
    return HttpResponse('<h1>Welcome to the GraphRAG Dashboard</h1>')
