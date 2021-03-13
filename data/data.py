"""
Michael Patel
March 2021

Project description:
    To classify NBA team logos

File description:
    To scrape NBA team logo images in order to build dataset

"""
################################################################################
# Imports
import os
import requests
import urllib3
from bs4 import BeautifulSoup


################################################################################
# Main
if __name__ == "__main__":
    # html soup
    url = "https://www.sportslogos.net/teams/list_by_league/6/National_Basketball_Association/NBA/logos/"
    http = urllib3.PoolManager()
    response = http.request("GET", url)
    page = response.data
    soup = BeautifulSoup(page, "html.parser")

    logo_grid = soup.find("div", {"id": "team"})
    hrefs = logo_grid.find_all("a")
    hrefs = hrefs[:30]
    #print(hrefs)
    #print(len(hrefs))
    #print(hrefs[0])

    teams = []
    image_urls = []

    for h in hrefs:
        name = " ".join(h.text.split())  # get team name

        # get team image url
        img = h.find("img")
        url = img["src"]

        # make directory named after team
        train_dir_path = os.path.join(os.getcwd(), "train")
        dir_path = os.path.join(train_dir_path, name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # save team image in team directory
        image_filename = name + ".jpg"
        image_filepath = os.path.join(dir_path, image_filename)

        with open(image_filepath, "wb") as f:
            response = requests.get(url, stream=True)

            if not response.ok:
                print(response)

            for block in response.iter_content(1024):
                if not block:
                    break

                f.write(block)

        teams.append(name)
        image_urls.append(url)
