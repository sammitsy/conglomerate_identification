from urllib.request import urlopen
import re
from bs4 import BeautifulSoup
from selenium import webdriver

import csv

def getURLlist(filename):
    ######
    # generate list of refcodes and urls from file
    # ##
    # Author: Nathan D. Gavin
    # Date: 24/3/23  
    # Description:
    # Function read .txt file containing all refcodes and urls and store in an array
    # ##
    # Input(s):
    # filename - name of .txt file containing refcodes and urls
    # ##
    # Output(s):
    # url_list - array of refcodes and corresponding urls
    # ##
    ######
    fid = open(filename,"r") # open .txt file
    lines = fid.readlines() # read each line of file
    fid.close() # close file
    nlines = len(lines) # total number of entries
    url_list = [] # create empty array to store refcodes and urls
    for line in range(nlines): # loop over each entry
        ln = lines[line] # get line
        lns = ln.split("\t") # split refcode and url (separated by tab)
        data = [] # create empty array for refcode and url
        for d in range(len(lns)): # loop over number of entries in line (expecting 2)
            data.append(lns[d].strip()) # remove whitespace and add to line data array
        url_list.append(data) # add line data to url list array
    return url_list

def getKEYWORDlist(filename):
    ######
    # generate list of keywords from file
    # ##
    # Author: Nathan D. Gavin
    # Date: 24/3/23  
    # Description:
    # Function read .txt file containing all keywords and store in an array
    # ##
    # Input(s):
    # filename - name of .txt file containing keywords
    # ##
    # Output(s):
    # keywords - array of keywords
    # ##
    ######
    fid = open(filename,"r") # open .txt file
    lines = fid.readlines() # get each line of file
    fid.close() # close file
    nk = len(lines) # number of keywords
    keywords = [] # create empty array
    for line in range(nk): # loop over each keyword
        ln = lines[line] # get keyword
        ln = ln.strip() # remove any whitespace
        keywords.append(ln) # add keyword to array
    return keywords

def getKeywordCount(text,keyword):
    ######
    # search text for a given keyword
    # ##
    # Author: Nathan D. Gavin
    # Date: 24/3/23  
    # Description:
    # Function to serach a text for a keyword and output the number of occurences of the keyword in the text (ignoring case)
    # ##
    # Input(s):
    # text - text to be searched
    # keyword - word or phrase which is to be searched for
    # ##
    # Output(s):
    # n_occur - number of occurences of the keyword in the text
    # ##
    ######
    n_occur = len(re.findall(keyword,text,re.IGNORECASE)) # find number of times keyword appears in text
    return n_occur

def runWebScrape(url_list,keywords,outputFilename):
    ######
    # scrape url list and search title and body for occurences of keywords
    # ##
    # Author: Nathan D. Gavin
    # Date: 24/3/23  
    # Description:
    # Function to open each url in a list, obtain the title and body of the html
    # and search the title and body texts to find out the number of times each keyword
    # in a given list appears in each text.
    # ##
    # Input(s):
    # url_list - array containing the each refcode and corresponding url
    # keywords - array containing each keyword to search for
    # outputFilename - filename of .csv file in which the data is to be written
    # ##
    # Output(s):
    # data - array containing all url data and number of occurences each keyword appears in the title and body of the url
    # ##
    ######

    nurl = len(url_list) # number of url entries to search
    ndp = 4  # number of entries of url data to store
    nk = len(keywords) # number of keywords to search
    ncol = ndp + 2 * nk # total number of columns in output file

    # create headers of output .csv file
    header0 = [""] * ncol 
    header0[ndp] = "title search"
    header0[ndp+nk] = "body search"

    header1 = [""] * ncol
    header1[0] = "index"
    header1[1] = "refcode"
    header1[2] = "DOI"
    header1[3] = "opened"
    header1[ndp:ndp+nk] = keywords
    header1[ndp+nk:] = keywords

    # create output .csv file
    with open(outputFilename, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # write headers to file
        writer.writerow(header0)
        writer.writerow(header1)
    
    # initialise data arrays
    data = [[""] * ncol for i in range(nurl)] # full data array
    urldata = [[""] * ndp for i in range(nurl)] # url data array
    titledata = [[0] * nk for i in range(nurl)] # title keyword search
    bodydata = [[0] * nk for i in range(nurl)] # body keyword search
    
    for i in range(nurl): # loop over each url
        refcode = url_list[i][0] # get refcode
        url = url_list[i][1] # get url

        urldata[i][0] = i # store index
        urldata[i][1] = refcode # store refcode
        urldata[i][2] = url # store url
        urldata[i][3] = "N" # initialise accessed state as false
        
        if url.casefold() == "none": # check if url is given
            print("{0} - no url given".format(i)) # print to terminal (url not opened as no url was given)
        else:
            try:
                dr = webdriver.Chrome() # create Chrome webdriver
                dr.get(url) # open url with Chrome
                urldata[i][3] = "Y" # set accessed state to true
                soup = BeautifulSoup(dr.page_source,"html.parser") # parse the html
                title = soup.title # get the page title
                title = title.get_text() # convert title html to text
                body = soup.body # get the page body
                body = body.get_text() # convert body html to text
                for j in range(nk): # loop over each keyword
                    kw = keywords[j] # get keyword from keyword list
                    titledata[i][j] = getKeywordCount(title,kw) # search title text for keyword
                    bodydata[i][j] = getKeywordCount(body,kw) # search body text for keyword
                print("{0} - DOI opened".format(i)) # print to terminal (url opened)
            except: # catch exceptions when opening url
                print("{0} - exception caught".format(i)) # print to terminal (not opened as error when opening)
    
        data[i][0:ndp] = urldata[i] # store url data
        data[i][ndp:ndp+nk] = titledata[i] # store title keyword search data
        data[i][ndp+nk:] = bodydata[i] # store body keyword search data

        # append data from url serach to output .csv file as new line
        with open(outputFilename, 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            # write data row
            writer.writerow(data[i])

    dr.quit() # kill the Chrome webdriver
        
    return data # return all data

urlFilename = "testing.txt" # file containing refcodes and urls (separated by a tab)
kwFilename = "keywordFile2.txt" # file containing keywords
outputFilename = "testing.csv" # filename where output is written

url_list = getURLlist(urlFilename) # get array of refcodes and urls
keywords = getKEYWORDlist(kwFilename) # get array of keywords
data = runWebScrape(url_list,keywords,outputFilename) # perform web scrape
