"""Models for ancient script recognition app."""
from django.db import models
from django.utils import timezone


class ScriptAnalysis(models.Model):
    """Store ancient script analysis results."""

    SCRIPT_TYPE_CHOICES = [
        ('甲骨文', '甲骨文'),
        ('敦煌文书', '敦煌文书'),
        ('金文', '金文'),
        ('篆书', '篆书'),
        ('隶书', '隶书'),
    ]

    image = models.ImageField(upload_to='uploads/%Y/%m/%d/')
    script_type = models.CharField(
        max_length=20,
        choices=SCRIPT_TYPE_CHOICES,
        default='甲骨文'
    )
    hint = models.TextField(blank=True, null=True)
    result = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = '古文字分析'
        verbose_name_plural = '古文字分析'

    def __str__(self):
        return f"{self.script_type} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"
