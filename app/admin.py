"""Django admin configuration."""
from django.contrib import admin
from .models import ScriptAnalysis


@admin.register(ScriptAnalysis)
class ScriptAnalysisAdmin(admin.ModelAdmin):
    list_display = ('id', 'script_type', 'created_at')
    list_filter = ('script_type', 'created_at')
    search_fields = ('hint', 'result')
    readonly_fields = ('created_at', 'updated_at')

    fieldsets = (
        ('基本信息', {
            'fields': ('image', 'script_type', 'hint')
        }),
        ('分析结果', {
            'fields': ('result',),
            'classes': ('collapse',)
        }),
        ('时间戳', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
