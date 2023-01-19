# This run_mmseqs2.py is copied from:
#    https://raw.githubusercontent.com/sokrypton/ColabFold/96fe2446f454eba38ea34ca45d97dc3f393e24ed/beta/colabfold.py
# and edited to just contain the code for run_mmseqs2
# imports
############################################
import requests
import time
import os
import random
import tarfile

def run_mmseqs2(x, prefix, use_env=True, use_filter=True,
                use_templates=False, filter=None, host_url="https://a3m.mmseqs.com"):

  def submit(seqs, mode, N=101):
    n,query = N,""
    for seq in seqs:
      query += ">{}\n{}\n".format(n, seq)
      n += 1

    res = requests.post('{}/ticket/msa'.format(host_url), data={'q':query,'mode': mode})
    try: out = res.json()
    except ValueError: out = {"status":"UNKNOWN"}
    return out

  def status(ID):
    res = requests.get('{}/ticket/{}'.format(host_url, ID))
    try: out = res.json()
    except ValueError: out = {"status":"UNKNOWN"}
    return out

  def download(ID, path):
    res = requests.get('{}/result/download/{}'.format(host_url, ID))
    with open(path,"wb") as out: out.write(res.content)

  # process input x
  seqs = [x] if isinstance(x, str) else x

  # compatibility to old option
  if filter is not None:
    use_filter = filter

  # setup mode
  if use_filter:
    mode = "env" if use_env else "all"
  else:
    mode = "env-nofilter" if use_env else "nofilter"

  # define path
  path = "{}_{}".format(prefix, mode)
  if not os.path.isdir(path): os.mkdir(path)

  # call mmseqs2 api
  tar_gz_file = '{}/out.tar.gz'.format(path)
  N,REDO = 101,True

  # deduplicate and keep track of order
  seqs_unique = sorted(list(set(seqs)))
  Ms = [N+seqs_unique.index(seq) for seq in seqs]

  # lets do it!
  if not os.path.isfile(tar_gz_file):
    TIME_ESTIMATE = 150 * len(seqs_unique)
    if 1:
      while REDO:

        # Resubmit job until it goes through
        out = submit(seqs_unique, mode, N)
        while out["status"] in ["UNKNOWN","RATELIMIT"]:
          # resubmit
          time.sleep(5 + random.randint(0,5))
          out = submit(seqs_unique, mode, N)

        if out["status"] == "ERROR":
          raise Exception('MMseqs2 API is giving errors. Please confirm your input is a valid protein sequence. If error persists, please try again an hour later.')

        if out["status"] == "MAINTENANCE":
          raise Exception('MMseqs2 API is undergoing maintenance. Please try again in a few minutes.')

        # wait for job to finish
        ID,TIME = out["id"],0
        while out["status"] in ["UNKNOWN","RUNNING","PENDING"]:
          t = 5 + random.randint(0,5)
          time.sleep(t)
          out = status(ID)
          if out["status"] == "RUNNING":
            TIME += t
          #if TIME > 900 and out["status"] != "COMPLETE":
          #  # something failed on the server side, need to resubmit
          #  N += 1
          #  break

        if out["status"] == "COMPLETE":
          if TIME < TIME_ESTIMATE:
            pass #
          REDO = False

      # Download results
      download(ID, tar_gz_file)

  # prep list of a3m files
  a3m_files = ["{}/uniref.a3m".format(path)]
  if use_env: a3m_files.append("{}/bfd.mgnify30.metaeuk30.smag30.a3m".format(path))

  # extract a3m files
  if not os.path.isfile(a3m_files[0]):
    with tarfile.open(tar_gz_file) as tar_gz:
      tar_gz.extractall(path)

  # templates
  if use_templates:
    templates = {}
    print("seq\tpdb\tcid\tevalue")
    for line in open("{}/pdb70.m8".format(path),"r"):
      p = line.rstrip().split()
      M,pdb,qid,e_value = p[0],p[1],p[2],p[10]
      M = int(M)
      if M not in templates: templates[M] = []
      templates[M].append(pdb)
      if len(templates[M]) <= 20:
        print("{}\t{}\t{}\t{}".format(int(M)-N, pdb, qid, e_value))

    template_paths = {}
    for k,TMPL in templates.items():
      TMPL_PATH = "{}_{}/templates_{}".format(prefix, mode, k)
      if not os.path.isdir(TMPL_PATH):
        os.mkdir(TMPL_PATH)
        TMPL_LINE = ",".join(TMPL[:20])
        os.system("curl -s https://a3m-templates.mmseqs.com/template/{} | tar xzf - -C {}/".format(TMPL_LINE, TMPL_PATH))
        os.system("cp {}/pdb70_a3m.ffindex {}/pdb70_cs219.ffindex".format(TMPL_PATH, TMPL_PATH))
        os.system("touch {}/pdb70_cs219.ffdata".format(TMPL_PATH))
      template_paths[k] = TMPL_PATH

  # gather a3m lines
  a3m_lines = {}
  for a3m_file in a3m_files:
    update_M,M = True,None
    for line in open(a3m_file,"r"):
      if len(line) > 0:
        if "\x00" in line:
          line = line.replace("\x00","")
          update_M = True
        if line.startswith(">") and update_M:
          M = int(line[1:].rstrip())
          update_M = False
          if M not in a3m_lines: a3m_lines[M] = []
        a3m_lines[M].append(line)

  # return results
  a3m_lines = ["".join(a3m_lines[n]) for n in Ms]

  if use_templates:
    template_paths_ = []
    for n in Ms:
      if n not in template_paths:
        template_paths_.append(None)
        print("{}\tno_templates_found",format(n-N))
      else:
        template_paths_.append(template_paths[n])
    template_paths = template_paths_

  if isinstance(x, str):
    return (a3m_lines[0], template_paths[0]) if use_templates else a3m_lines[0]
  else:
    return (a3m_lines, template_paths) if use_templates else a3m_lines
