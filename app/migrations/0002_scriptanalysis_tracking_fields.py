from django.db import migrations, models


def _add_missing_columns(apps, schema_editor):
    """Ensure legacy databases gain the tracking columns before state update."""
    ScriptAnalysis = apps.get_model('app', 'ScriptAnalysis')
    table = ScriptAnalysis._meta.db_table
    connection = schema_editor.connection
    with connection.cursor() as cursor:
        existing_columns = {
            column.name for column in connection.introspection.get_table_description(cursor, table)
        }

    def ensure_field(field_name: str, field: models.Field) -> None:
        if field_name in existing_columns:
            return
        field.set_attributes_from_name(field_name)
        schema_editor.add_field(ScriptAnalysis, field)

    ensure_field('num_references', models.PositiveIntegerField(default=0))
    ensure_field('rag_references', models.TextField(blank=True, null=True))
    ensure_field('rag_scores', models.TextField(blank=True, null=True))
    ensure_field('rag_text_info', models.TextField(blank=True, null=True))


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0001_initial'),
    ]

    operations = [
        migrations.SeparateDatabaseAndState(
            database_operations=[
                migrations.RunPython(_add_missing_columns,
                                     migrations.RunPython.noop),
            ],
            state_operations=[
                migrations.AddField(
                    model_name='scriptanalysis',
                    name='num_references',
                    field=models.PositiveIntegerField(
                        default=0,
                        help_text='Number of contextual references used during analysis'
                    ),
                ),
                migrations.AddField(
                    model_name='scriptanalysis',
                    name='rag_references',
                    field=models.TextField(
                        blank=True,
                        null=True,
                        help_text='Serialized reference identifiers returned by the RAG pipeline'
                    ),
                ),
                migrations.AddField(
                    model_name='scriptanalysis',
                    name='rag_scores',
                    field=models.TextField(
                        blank=True,
                        null=True,
                        help_text='Serialized similarity scores for each retrieved reference'
                    ),
                ),
                migrations.AddField(
                    model_name='scriptanalysis',
                    name='rag_text_info',
                    field=models.TextField(
                        blank=True,
                        null=True,
                        help_text='Serialized textual snippets retrieved alongside the image context'
                    ),
                ),
            ],
        ),
    ]
