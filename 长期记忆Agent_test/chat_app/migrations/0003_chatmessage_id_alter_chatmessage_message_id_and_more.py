# Generated by Django 5.2.1 on 2025-05-25 13:58

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("chat_app", "0002_remove_chatmessage_id_remove_customuser_bio_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="chatmessage",
            name="id",
            field=models.AutoField(primary_key=True, serialize=False),
        ),
        migrations.AlterField(
            model_name="chatmessage",
            name="message_id",
            field=models.CharField(
                default="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                editable=False,
                max_length=32,
                unique=True,
            ),
        ),
        migrations.AlterModelTable(
            name="chatmessage",
            table="chat_app_chatmessage",
        ),
    ]
