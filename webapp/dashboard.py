from flask import Markup

core = """<div style="width: 90%;height:100px;border-radius: 20px;box-shadow: 0px 0px 20px rgba(0, 0, 0, 0.2);margin-left: 5%;margin-right: 5%;margin-top: 30px;display: flex;align-items: center;">
        <div class="w3-row" style="width: 100%;text-align: center;">
            <div class="w3-container w3-quarter" style="border-right: 1px solid rgba(128, 128, 128, 0.472);display: flex;align-items:center;justify-content: center;">
              {status}
            </div>
            <div class="w3-container w3-quarter" style="border-right: 1px solid rgba(128, 128, 128, 0.472);">
              <h2><span style="color: #4866E6;">{date}</span></h2>
            </div>
            <div class="w3-container w3-quarter" style="border-right: 1px solid rgba(128, 128, 128, 0.472);">
              <h2><span style="color: #4866E6;">{elements}</span> elements</h2>
            </div>
            <div class="w3-container w3-quarter">
              <h2 style="color: #4866E6;">Review</h2>
            </div>
          </div>
    </div>"""

queued = """<img src="{image}" style="height: 40px;vertical-align: middle;">
              <h2 style="color: #FDDC27;margin-left: 20px;">Queued</h2>""".format(image='../static/img/queued.svg')

trained = """<img src="{image}" style="height: 40px;vertical-align: middle;">
              <h2 style="color: #4BD184;margin-left: 20px;">Trained</h2>""".format(image='../static/img/trained.svg')

class Dashboard:
    def __init__(self, user):
        super().__init__()
        self.user = user
        self.weeks = 2
        self.latestUpdate = "2 weeks"
        self.accuracy = 80

        # this is an example showing the encoding of the receipts. This should be provided by the database.
        self.receipts = [
            {
                "id": 1,
                "status": "queued",
                "date": "29.11.2020",
                "elements": 32,
                "url" : "NULL"
            },
            {
                "id": 2,
                "status": "trained",
                "date": "23.11.2020",
                "elements": 21,
                "url" : "NULL"
            },
            {
                "id": 3,
                "status": "trained",
                "date": "20.11.2020",
                "elements": 11,
                "url" : "NULL"
            },
            {
                "id": 4,
                "status": "trained",
                "date": "15.11.2020",
                "elements": 42,
                "url" : "NULL"
            },
            {
                "id": 5,
                "status": "trained",
                "date": "10.11.2020",
                "elements": 12,
                "url" : "NULL"
            },
        ]
        # TODO
        # this is an example showing the encoding of the list. This should also be provided by the database.
        self.lists = []

    def generateHTMLForReceipts(self):
        html = ""
        for receipt in self.receipts:
            if receipt["status"] == "trained":
                status = trained
            else:
                status = queued
            html += core.format(status=status, date=receipt["date"], elements=receipt["elements"])
        return Markup(html)
    