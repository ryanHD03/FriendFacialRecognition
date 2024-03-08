from CNN import CNN_model, test_image
from image_manipulation import resize
import discord
from discord.ext import commands
import uuid
import requests
import shutil
import os

intents = discord.Intents.default()
intents.message_content = True
client = commands.Bot(command_prefix= '.', intents = intents)

@client.event
async def on_ready():
    print("Bot is up and running!")
@client.command()
async def save(ctx):
    try:
        url = ctx.message.attachments[0].url
    except IndexError:
        print("Error: No attachments")
        await ctx.send("No attachments detected. Please try again.")
    else:
        if url[0:26] == "https://cdn.discordapp.com":
            r = requests.get(url, stream = True)
            imageName = str(uuid.uuid4()) + '.png'
            folder_path = "C:\Ryan\PersonalProject\\FriendRecog\\bot\images"
            full_path = os.path.join(folder_path, imageName)
            with open(full_path, 'wb') as file:
                print('Saving Image: ' + imageName)
                shutil.copyfileobj(r.raw, file)

            resize()
            model = CNN_model()
            predicted_class, confidence = test_image(model)
            await ctx.send("Prediction: "  + str(predicted_class) + " with a " + 100 * str(confidence) + "% confidence level.")