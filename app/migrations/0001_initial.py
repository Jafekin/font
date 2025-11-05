from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='ScriptAnalysis',
            fields=[
                ('id', models.BigAutoField(auto_created=True,
                 primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.ImageField(upload_to='uploads/%Y/%m/%d/')),
                ('script_type', models.CharField(choices=[('甲骨文', '甲骨文'), ('敦煌文书', '敦煌文书'), (
                    '金文', '金文'), ('篆书', '篆书'), ('隶书', '隶书')], default='甲骨文', max_length=20)),
                ('hint', models.TextField(blank=True, null=True)),
                ('result', models.TextField()),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
            ],
            options={
                'verbose_name': '古文字分析',
                'verbose_name_plural': '古文字分析',
                'ordering': ['-created_at'],
            },
        ),
    ]
