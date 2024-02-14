# This run_mmseqs2.py is copied from:
# https://github.com/sokrypton/ColabFold/blob/main/colabfold/colabfold.py
# and edited to just contain the code for run_mmseqs2 and to allow
# skipping MSA and saving the PDB file names
# imports
############################################
import requests
import time
import os, sys
import random
import tarfile

def run_mmseqs2(x, prefix, use_env=True, use_filter=True,
                use_templates=False,
                skip_msa=False,
                pdb70_text="",
                filter=None, use_pairing=False,
                pairing_strategy="greedy",
                host_url="https://api.colabfold.com",
                user_agent="phenix/user", log = sys.stdout):

  submission_endpoint = "ticket/pair" if use_pairing else "ticket/msa"

  headers = {}
  if user_agent != "":
    headers['User-Agent'] = user_agent
  else:
    print("No user agent specified. Please set a user agent (e.g., 'toolname/version contact@email') to help us debug in case of problems. This warning will become an error in the future.", file = log)

  def submit(seqs, mode, N=101):
    n, query = N, ""
    for seq in seqs:
      query += f">{n}\n{seq}\n"
      n += 1

    while True:
      error_count = 0
      try:
        # https://requests.readthedocs.io/en/latest/user/advanced/#advanced
        # "good practice to set connect timeouts to slightly larger than a multiple of 3"
        res = requests.post(f'{host_url}/{submission_endpoint}', data={ 'q': query, 'mode': mode }, timeout=6.02, headers=headers)
      except requests.exceptions.Timeout:
        print("Timeout while submitting to MSA server. Retrying...", file = log)
        continue
      except Exception as e:
        error_count += 1
        print(f"Error while fetching result from MSA server. Retrying... ({error_count}/5)", file = log)
        print(f"Error: {e}", file = log)
        time.sleep(5)
        if error_count > 5:
          raise
        continue
      break

    try:
      out = res.json()
    except ValueError:
      print(f"Server didn't reply with json: {res.text}", file = log)
      out = {"status":"ERROR"}
    return out

  def status(ID):
    while True:
      error_count = 0
      try:
        res = requests.get(f'{host_url}/ticket/{ID}', timeout=6.02, headers=headers)
      except requests.exceptions.Timeout:
        print("Timeout while fetching status from MSA server. Retrying...",
          file = log)
        continue
      except Exception as e:
        error_count += 1
        print(f"Error while fetching result from MSA server. Retrying... ({error_count}/5)", file = log)
        print(f"Error: {e}", file = log)
        time.sleep(5)
        if error_count > 5:
          raise
        continue
      break
    try:
      out = res.json()
    except ValueError:
      print(f"Server didn't reply with json: {res.text}", file = log)
      out = {"status":"ERROR"}
    return out

  def download(ID, path):
    error_count = 0
    while True:
      try:
        res = requests.get(f'{host_url}/result/download/{ID}', timeout=6.02, headers=headers)
      except requests.exceptions.Timeout:
        print("Timeout while fetching result from MSA server. Retrying...",
          file = log)
        continue
      except Exception as e:
        error_count += 1
        print(f"Error while fetching result from MSA server. Retrying... ({error_count}/5)",  file = log)
        print(f"Error: {e}", file = log)
        time.sleep(5)
        if error_count > 5:
          raise
        continue
      break
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

  if use_pairing:
    use_templates = False
    use_env = False
    mode = ""
    # greedy is default, complete was the previous behavior
    if pairing_strategy == "greedy":
      mode = "pairgreedy"
    elif pairing_strategy == "complete":
      mode = "paircomplete"

  # define path
  path = f"{prefix}_{mode}"
  if not os.path.isdir(path): os.mkdir(path)

  # call mmseqs2 api
  tar_gz_file = f'{path}/out.tar.gz'
  N,REDO = 101,True

  # deduplicate and keep track of order
  seqs_unique = []
  #TODO this might be slow for large sets
  [seqs_unique.append(x) for x in seqs if x not in seqs_unique]
  Ms = [N + seqs_unique.index(seq) for seq in seqs]
  # lets do it!
  if not os.path.isfile(tar_gz_file):
    TIME_ESTIMATE = 150 * len(seqs_unique)
    while REDO:

        # Resubmit job until it goes through
        out = submit(seqs_unique, mode, N)
        while out["status"] in ["UNKNOWN", "RATELIMIT"]:
          sleep_time = 5 + random.randint(0, 5)
          print(f"Sleeping for {sleep_time}s. Reason: {out['status']}", file = log)
          # resubmit
          time.sleep(sleep_time)
          out = submit(seqs_unique, mode, N)

        if out["status"] == "ERROR":
          raise Exception(f'MMseqs2 API is giving errors. Please confirm your input is a valid protein sequence. If error persists, please try again an hour later.')

        if out["status"] == "MAINTENANCE":
          raise Exception(f'MMseqs2 API is undergoing maintenance. Please try again in a few minutes.')

        # wait for job to finish
        ID,TIME = out["id"],0
        while out["status"] in ["UNKNOWN","RUNNING","PENDING"]:
          t = 5 + random.randint(0,5)
          print(f"Sleeping for {t}s. Reason: {out['status']}", file = log)
          time.sleep(t)
          out = status(ID)
          if out["status"] == "RUNNING":
            TIME += t
          #if TIME > 900 and out["status"] != "COMPLETE":
          #  # something failed on the server side, need to resubmit
          #  N += 1
          #  break

        if out["status"] == "COMPLETE":
          REDO = False

        if out["status"] == "ERROR":
          REDO = False
          raise Exception(f'MMseqs2 API is giving errors. Please confirm your input is a valid protein sequence. If error persists, please try again an hour later.')

    # Download results
    download(ID, tar_gz_file)

  # prep list of a3m files
  if use_pairing:
    a3m_files = [f"{path}/pair.a3m"]
  else:
    a3m_files = [f"{path}/uniref.a3m"]
    if use_env: a3m_files.append(f"{path}/bfd.mgnify30.metaeuk30.smag30.a3m")

  # extract a3m files
  if any(not os.path.isfile(a3m_file) for a3m_file in a3m_files):
    with tarfile.open(tar_gz_file) as tar_gz:
      tar_gz.extractall(path)

  # templates
  fn = f"{path}/pdb70.m8"
  if fn and os.path.isfile(fn) and (not pdb70_text):
    pdb70_text = open(fn).read()

  if use_templates:
    templates = {}
    for line in pdb70_text.splitlines():
      p = line.rstrip().split()
      M,pdb,qid,e_value = p[0],p[1],p[2],p[10]
      M = int(M)
      if M not in templates: templates[M] = []
      templates[M].append(pdb)
      #if len(templates[M]) <= 20:
      #  print(f"{int(M)-N}\t{pdb}\t{qid}\t{e_value}", file = log)

    template_paths = {}
    for k,TMPL in templates.items():
      TMPL_PATH = f"{prefix}_{mode}/templates_{k}"
      if not os.path.isdir(TMPL_PATH):
        os.mkdir(TMPL_PATH)
        TMPL_LINE = ",".join(TMPL[:20])
        response = None
        while True:
          error_count = 0
          try:
            # https://requests.readthedocs.io/en/latest/user/advanced/#advanced
            # "good practice to set connect timeouts to slightly larger than a multiple of 3"
            response = requests.get(f"{host_url}/template/{TMPL_LINE}", stream=True, timeout=6.02, headers=headers)
          except requests.exceptions.Timeout:
            print("Timeout while submitting to template server. Retrying...",
              file = log)
            continue
          except Exception as e:
            error_count += 1
            print(f"Error while fetching result from template server. Retrying... ({error_count}/5)", file = log)
            print(f"Error: {e}", file = log)
            time.sleep(5)
            if error_count > 5:
              raise
            continue
          break
        with tarfile.open(fileobj=response.raw, mode="r|gz") as tar:
          tar.extractall(path=TMPL_PATH)
        os.symlink("pdb70_a3m.ffindex", f"{TMPL_PATH}/pdb70_cs219.ffindex")
        with open(f"{TMPL_PATH}/pdb70_cs219.ffdata", "w") as f:
          f.write("")
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
        #print(f"{n-N}\tno_templates_found", file = log)
      else:
        template_paths_.append(template_paths[n])
    template_paths = template_paths_

  if isinstance(x, str):
    return (a3m_lines[0], template_paths[0], pdb70_text) if use_templates else (
    a3m_lines[0], pdb70_text)
  else:
    return (a3m_lines, template_paths, pdb70_text) if use_templates else (
      a3m_lines, pdb70_text)

